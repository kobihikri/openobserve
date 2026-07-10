// Copyright 2026 OpenObserve Inc.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

//! Tantivy warm-up plan construction.
//!
//! This module only extracts the warm-up logic that previously lived in
//! `search_tantivy_index`; it intentionally preserves the existing full-field,
//! exact-term, and fast-field warm-up behavior.

use std::collections::HashSet;

use config::{TIMESTAMP_COL_NAME, meta::inverted_index::IndexOptimizeMode};
use hashbrown::HashMap;
use tantivy::{
    SegmentReader, Term,
    query::Query,
    schema::{Field, Schema},
};
use tantivy_utils::puffin_directory::reader::warm_up_terms;

use crate::service::search::index::IndexCondition;

#[derive(Default)]
pub(super) struct WarmPlan {
    terms: HashMap<Field, HashMap<Term, bool>>,
    full_posting_fields: HashSet<Field>,
    fast_fields: HashSet<String>,
}

impl WarmPlan {
    pub(super) fn build(
        condition: &IndexCondition,
        query: &dyn Query,
        idx_optimize_rule: &Option<IndexOptimizeMode>,
        schema: &Schema,
        file_in_range: bool,
        has_skipped_conditions: bool,
    ) -> Self {
        let full_posting_fields = condition
            .need_all_term_fields()
            .into_iter()
            .chain(simple_distinct_field(idx_optimize_rule))
            .filter_map(|field| schema.get_field(&field).ok())
            .collect();

        let mut terms: HashMap<Field, HashMap<Term, bool>> = HashMap::new();
        query.query_terms(&mut |term, need_position| {
            terms
                .entry(term.field())
                .or_default()
                .insert(term.clone(), need_position);
        });

        let fast_fields = fast_fields(idx_optimize_rule, file_in_range, has_skipped_conditions);

        Self {
            terms,
            full_posting_fields,
            fast_fields,
        }
    }

    pub(super) async fn execute(self, segment_reader: &SegmentReader) -> anyhow::Result<()> {
        warm_up_terms(
            segment_reader,
            &self.terms,
            self.full_posting_fields,
            self.fast_fields,
        )
        .await
    }
}

fn simple_distinct_field(
    idx_optimize_rule: &Option<IndexOptimizeMode>,
) -> impl Iterator<Item = String> {
    match idx_optimize_rule {
        Some(IndexOptimizeMode::SimpleDistinct(field, ..)) => Some(field.clone()).into_iter(),
        _ => None.into_iter(),
    }
}

fn fast_fields(
    idx_optimize_rule: &Option<IndexOptimizeMode>,
    file_in_range: bool,
    has_skipped_conditions: bool,
) -> HashSet<String> {
    let mut fields = HashSet::new();
    if let Some(rule) = idx_optimize_rule {
        match rule {
            IndexOptimizeMode::SimpleHistogram(..) => {
                fields.insert(TIMESTAMP_COL_NAME.to_string());
            }
            IndexOptimizeMode::SimpleMultiHistogram(.., name) => {
                fields.insert(TIMESTAMP_COL_NAME.to_string());
                fields.insert(name.clone());
            }
            IndexOptimizeMode::SimpleTopN(names, ..) => {
                fields.extend(names.iter().cloned());
            }
            IndexOptimizeMode::SimpleSelect(..) if !has_skipped_conditions => {
                fields.insert(TIMESTAMP_COL_NAME.to_string());
            }
            _ => {}
        }
    }
    if !file_in_range {
        fields.insert(TIMESTAMP_COL_NAME.to_string());
    }
    fields
}

#[cfg(test)]
mod tests {
    use config::INDEX_FIELD_NAME_FOR_ALL;
    use tantivy::schema::{FAST, Schema, TEXT};

    use super::*;
    use crate::service::search::index::Condition;

    fn schema() -> Schema {
        let mut builder = Schema::builder();
        builder.add_text_field("tag", TEXT);
        builder.add_text_field(INDEX_FIELD_NAME_FOR_ALL, TEXT);
        builder.add_i64_field(TIMESTAMP_COL_NAME, FAST);
        builder.build()
    }

    fn build_plan(
        condition: Condition,
        optimize_rule: Option<IndexOptimizeMode>,
        file_in_range: bool,
        has_skipped_conditions: bool,
    ) -> (WarmPlan, Schema) {
        let schema = schema();
        let default_field = schema.get_field(INDEX_FIELD_NAME_FOR_ALL).ok();
        let mut index_condition = IndexCondition::new();
        index_condition.add_condition(condition);
        let (query, _) = index_condition
            .to_tantivy_query("test", schema.clone(), default_field)
            .unwrap();
        let plan = WarmPlan::build(
            &index_condition,
            query.as_ref(),
            &optimize_rule,
            &schema,
            file_in_range,
            has_skipped_conditions,
        );
        (plan, schema)
    }

    #[test]
    fn simple_distinct_field_preserves_existing_mode_selection() {
        assert_eq!(
            simple_distinct_field(&None).collect::<Vec<_>>(),
            Vec::<String>::new()
        );
        assert_eq!(
            simple_distinct_field(&Some(IndexOptimizeMode::SimpleCount)).collect::<Vec<_>>(),
            Vec::<String>::new()
        );
        assert_eq!(
            simple_distinct_field(&Some(IndexOptimizeMode::SimpleDistinct(
                "tag".into(),
                10,
                true,
            )))
            .collect::<Vec<_>>(),
            vec!["tag".to_string()]
        );
    }

    #[test]
    fn str_match_preserves_full_posting_warmup() {
        let (plan, schema) = build_plan(
            Condition::StrMatch("tag".into(), "needle".into(), true),
            None,
            true,
            false,
        );
        assert_eq!(
            plan.full_posting_fields,
            HashSet::from([schema.get_field("tag").unwrap()])
        );
        assert!(plan.terms.is_empty());
    }

    #[test]
    fn not_equal_preserves_full_field_and_exact_term_warmup() {
        let (plan, schema) = build_plan(
            Condition::NotEqual("tag".into(), "value".into()),
            None,
            true,
            false,
        );
        let tag = schema.get_field("tag").unwrap();
        assert_eq!(plan.full_posting_fields, HashSet::from([tag]));
        assert_eq!(
            plan.terms.get(&tag),
            Some(&HashMap::from([(
                Term::from_field_text(tag, "value"),
                false
            )]))
        );
    }

    #[test]
    fn simple_distinct_preserves_target_field_warmup() {
        let (plan, schema) = build_plan(
            Condition::All(),
            Some(IndexOptimizeMode::SimpleDistinct("tag".into(), 10, true)),
            true,
            false,
        );
        assert_eq!(
            plan.full_posting_fields,
            HashSet::from([schema.get_field("tag").unwrap()])
        );
    }

    #[test]
    fn collector_fast_fields_are_preserved() {
        let (histogram, _) = build_plan(
            Condition::All(),
            Some(IndexOptimizeMode::SimpleHistogram(0, 100, 10, 0)),
            true,
            false,
        );
        assert_eq!(
            histogram.fast_fields,
            HashSet::from([TIMESTAMP_COL_NAME.to_string()])
        );

        let (top_n, _) = build_plan(
            Condition::All(),
            Some(IndexOptimizeMode::SimpleTopN(vec!["tag".into()], 10, true)),
            true,
            false,
        );
        assert_eq!(top_n.fast_fields, HashSet::from(["tag".to_string()]));
    }

    #[test]
    fn partial_file_preserves_timestamp_warmup() {
        let (plan, _) = build_plan(Condition::All(), None, false, false);
        assert_eq!(
            plan.fast_fields,
            HashSet::from([TIMESTAMP_COL_NAME.to_string()])
        );
    }

    #[test]
    fn skipped_simple_select_preserves_existing_fast_field_behavior() {
        let (skipped, _) = build_plan(
            Condition::All(),
            Some(IndexOptimizeMode::SimpleSelect(10, true)),
            true,
            true,
        );
        assert!(skipped.fast_fields.is_empty());

        let (complete, _) = build_plan(
            Condition::All(),
            Some(IndexOptimizeMode::SimpleSelect(10, true)),
            true,
            false,
        );
        assert_eq!(
            complete.fast_fields,
            HashSet::from([TIMESTAMP_COL_NAME.to_string()])
        );
    }
}
