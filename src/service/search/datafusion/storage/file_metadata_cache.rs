// Copyright 2026 OpenObserve Inc.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, LazyLock as Lazy,
        atomic::{AtomicI64, Ordering},
    },
    time::Instant,
};

use config::metrics;
use dashmap::DashMap;
use datafusion::execution::cache::{
    CacheAccessor,
    cache_manager::{self, CachedFileMetadataEntry, FileMetadata, FileMetadataCacheEntry},
};
use object_store::{ObjectMeta, path::Path};

use super::TRACE_ID_SEPARATOR;

pub static GLOBAL_CACHE: Lazy<Arc<FileMetadataCache>> =
    Lazy::new(|| Arc::new(FileMetadataCache::default()));

/// Process-wide cache of decoded parquet footers (`ParquetMetaData`).
///
/// DataFusion's `CachedParquetFileReaderFactory` consults this cache through the
/// runtime `CacheManager` before fetching/decoding a file's footer, so a hit
/// skips both the footer range reads and the thrift decode on every scan.
/// Entries are validated by the caller via `CachedFileMetadataEntry::is_valid_for`
/// (size + last_modified), which are stable per file in our object stores.
pub struct FileMetadataCache {
    metadata: DashMap<String, (ObjectMeta, Arc<dyn FileMetadata>, usize)>,
    cacher: parking_lot::Mutex<VecDeque<String>>,
    current_memory: AtomicI64,
}

impl FileMetadataCache {
    pub fn new() -> Self {
        Self {
            metadata: DashMap::new(),
            cacher: parking_lot::Mutex::new(VecDeque::new()),
            current_memory: AtomicI64::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    pub fn memory_size(&self) -> usize {
        self.current_memory.load(Ordering::Relaxed).max(0) as usize
    }

    fn estimate_entry_size(key: &str, meta: &ObjectMeta, value: &dyn FileMetadata) -> usize {
        // Key is stored both in the DashMap and the eviction queue
        let mut size = (std::mem::size_of::<String>() + key.len()) * 2;

        size += std::mem::size_of::<ObjectMeta>();
        size += meta.location.as_ref().len();
        if let Some(ref etag) = meta.e_tag {
            size += std::mem::size_of::<String>() + etag.len();
        }
        if let Some(ref version) = meta.version {
            size += std::mem::size_of::<String>() + version.len();
        }

        // The decoded metadata itself dominates: for parquet this is
        // `ParquetMetaData::memory_size()`, including page indexes when present.
        size += value.memory_size();
        size += std::mem::size_of::<usize>(); // tracked entry size field
        // Arc header (strong + weak counter) and DashMap bucket overhead
        // (hash + slot metadata). Conservative fixed overhead per entry so
        // the memory budget is not systematically underestimated.
        size += 64;

        size
    }

    fn evict(&self, max_bytes: usize) {
        let start = Instant::now();
        let mut warned = false;
        let mut w = self.cacher.lock();
        while self.current_memory.load(Ordering::Relaxed) > max_bytes as i64 && !w.is_empty() {
            if !warned {
                log::warn!(
                    "FileMetadataCache is full ({} bytes > {} bytes), evicting oldest entries",
                    self.current_memory.load(Ordering::Relaxed),
                    max_bytes,
                );
                warned = true;
            }
            let batch = (w.len() / 20).max(1).min(w.len());
            let mut removed_total = 0i64;
            for k in w.drain(0..batch) {
                if let Some((_, (_, _, size))) = self.metadata.remove(&k) {
                    removed_total += size as i64;
                }
            }
            if removed_total > 0 {
                self.current_memory
                    .fetch_sub(removed_total, Ordering::Relaxed);
            }
        }
        drop(w);
        metrics::QUERY_PARQUET_FILE_METADATA_CACHE_GC_COUNT
            .with_label_values::<&str>(&[])
            .inc();
        metrics::QUERY_PARQUET_FILE_METADATA_CACHE_GC_TIME
            .with_label_values::<&str>(&[])
            .observe(start.elapsed().as_millis() as f64);
    }

    /// Strip the per-query trace-id/schema prefix so the same physical file
    /// keys consistently across queries (mirrors `FileStatisticsCache`).
    fn format_key(&self, k: &Path) -> String {
        if let Some(mut p) = k.as_ref().find(TRACE_ID_SEPARATOR) {
            if let Some(pp) = k.as_ref()[..p].find("/schema=") {
                p = pp;
            }
            k.as_ref()[p..].to_string()
        } else {
            k.to_string()
        }
    }
}

impl Default for FileMetadataCache {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheAccessor<Path, CachedFileMetadataEntry> for FileMetadataCache {
    /// Get the cached decoded metadata for a file location.
    fn get(&self, k: &Path) -> Option<CachedFileMetadataEntry> {
        let k = self.format_key(k);
        match self.metadata.get(&k) {
            Some(entry) => {
                metrics::QUERY_PARQUET_FILE_METADATA_CACHE_HITS_TOTAL
                    .with_label_values::<&str>(&[])
                    .inc();
                let (meta, value, _) = entry.value();
                Some(CachedFileMetadataEntry::new(meta.clone(), value.clone()))
            }
            None => {
                metrics::QUERY_PARQUET_FILE_METADATA_CACHE_MISS_TOTAL
                    .with_label_values::<&str>(&[])
                    .inc();
                None
            }
        }
    }

    /// Save the decoded metadata for a file location.
    fn put(&self, k: &Path, value: CachedFileMetadataEntry) -> Option<CachedFileMetadataEntry> {
        let k = self.format_key(k);
        let entry_size = Self::estimate_entry_size(&k, &value.meta, value.file_metadata.as_ref());

        let old = self.metadata.insert(
            k.clone(),
            (value.meta.clone(), value.file_metadata.clone(), entry_size),
        );
        let old_size = old.as_ref().map(|(_, _, s)| *s).unwrap_or(0);
        let delta = entry_size as i64 - old_size as i64;
        self.current_memory.fetch_add(delta, Ordering::Relaxed);

        // Only queue the key for eviction when it's a fresh insertion.
        // When `old.is_some()` the key is already present in `cacher`
        // (it hasn't been drained yet, otherwise metadata wouldn't hold it),
        // so pushing again would create a duplicate tombstone.
        if old.is_none() {
            self.cacher.lock().push_back(k);
        }

        let max_bytes = config::get_config()
            .limit
            .datafusion_file_metadata_cache_max_size;
        if self.current_memory.load(Ordering::Relaxed) > max_bytes as i64 {
            self.evict(max_bytes);
        }

        old.map(|(meta, value, _)| CachedFileMetadataEntry::new(meta, value))
    }

    fn remove(&self, k: &Path) -> Option<CachedFileMetadataEntry> {
        let k = self.format_key(k);
        self.metadata.remove(&k).map(|(_, (meta, value, size))| {
            self.current_memory
                .fetch_sub(size as i64, Ordering::Relaxed);
            CachedFileMetadataEntry::new(meta, value)
        })
    }

    fn contains_key(&self, k: &Path) -> bool {
        let k = self.format_key(k);
        self.metadata.contains_key(&k)
    }

    fn len(&self) -> usize {
        self.metadata.len()
    }

    fn clear(&self) {
        self.metadata.clear();
        self.cacher.lock().clear();
        self.current_memory.store(0, Ordering::Relaxed);
    }

    fn name(&self) -> String {
        "FileMetadataCache".to_string()
    }
}

impl cache_manager::FileMetadataCache for FileMetadataCache {
    fn cache_limit(&self) -> usize {
        config::get_config()
            .limit
            .datafusion_file_metadata_cache_max_size
    }

    fn update_cache_limit(&self, _limit: usize) {
        // No-op: this is a process-wide, long-lived global cache whose eviction
        // budget is owned solely by `put()` via
        // `datafusion_file_metadata_cache_max_size`. DataFusion calls this once
        // per `CacheManager::try_new` (i.e. per session / per query) with its
        // own per-session limit.
    }

    fn list_entries(&self) -> HashMap<Path, FileMetadataCacheEntry> {
        let mut entries = HashMap::<Path, FileMetadataCacheEntry>::new();

        for entry in &self.metadata {
            let path = Path::from(entry.key().as_str());
            let (object_meta, value, size) = entry.value();
            entries.insert(
                path,
                FileMetadataCacheEntry {
                    object_meta: object_meta.clone(),
                    size_bytes: *size,
                    hits: 0,
                    extra: value.extra_info(),
                },
            );
        }

        entries
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use chrono::{DateTime, Utc};

    use super::*;

    /// Minimal [`FileMetadata`] implementation with a fixed reported size.
    struct TestMetadata {
        size: usize,
    }

    impl FileMetadata for TestMetadata {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn memory_size(&self) -> usize {
            self.size
        }

        fn extra_info(&self) -> HashMap<String, String> {
            HashMap::new()
        }
    }

    /// Parse an RFC3339 timestamp, panicking on malformed input (test-only).
    fn ts(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().into()
    }

    /// Minimal [`ObjectMeta`] for a location, fixed `last_modified`, no etag/version.
    fn object_meta(location: &str, size: u64) -> ObjectMeta {
        ObjectMeta {
            location: Path::from(location),
            last_modified: ts("2024-01-15T00:00:00+00:00"),
            size,
            e_tag: None,
            version: None,
        }
    }

    /// Insert an entry into the cache under its own location.
    fn put(cache: &FileMetadataCache, meta: ObjectMeta, size: usize) {
        cache.put(
            &meta.location.clone(),
            CachedFileMetadataEntry::new(meta, Arc::new(TestMetadata { size })),
        );
    }

    #[test]
    fn test_file_metadata_cache() {
        let meta = object_meta("files/default/logs/test.parquet", 1024);
        let cache = FileMetadataCache::default();
        assert!(cache.get(&meta.location).is_none());

        put(&cache, meta.clone(), 100);

        // exact match is valid
        let cached = cache.get(&meta.location).expect("entry present");
        assert!(cached.is_valid_for(&meta));

        // same location but file size changed -> cached but stale
        let mut changed = meta.clone();
        changed.size = 2048;
        let cached = cache.get(&changed.location).expect("entry present");
        assert!(!cached.is_valid_for(&changed));

        // same location but last_modified changed -> cached but stale
        let mut changed = meta.clone();
        changed.last_modified = ts("2024-01-15T01:00:00+00:00");
        let cached = cache.get(&changed.location).expect("entry present");
        assert!(!cached.is_valid_for(&changed));

        // different location -> miss
        assert!(cache.get(&Path::from("files/other.parquet")).is_none());
    }

    #[test]
    fn test_trace_id_prefix_normalization() {
        // The same physical file scanned by two different queries (different
        // trace-id prefixes) must share one cache entry.
        let cache = FileMetadataCache::new();
        let physical = "$$/files/default/logs/test.parquet";
        let meta = object_meta(
            &format!("trace1/schema=abc/format=parquet/{physical}"),
            1024,
        );
        put(&cache, meta, 100);

        assert_eq!(cache.len(), 1);
        let other_trace = Path::from(format!("trace2/schema=abc/format=parquet/{physical}"));
        assert!(cache.get(&other_trace).is_some());
        assert!(cache.contains_key(&other_trace));
    }

    #[test]
    fn test_memory_size_tracking() {
        let cache = FileMetadataCache::new();
        assert_eq!(cache.memory_size(), 0, "empty cache tracks zero memory");

        for i in 0..10u64 {
            let meta = object_meta(&format!("files/test_file_{i}"), 1024 * (i + 1));
            put(&cache, meta, 1000);
        }

        assert!(
            cache.memory_size() >= 10_000,
            "memory includes the reported metadata sizes"
        );
        assert_eq!(cache.len(), 10);

        // overwriting a key must not double-count
        let before = cache.memory_size();
        let meta = object_meta("files/test_file_0", 1024);
        put(&cache, meta, 1000);
        assert_eq!(cache.len(), 10);
        assert_eq!(cache.memory_size(), before);

        cache.clear();
        assert_eq!(cache.memory_size(), 0, "cleared cache returns to zero");
    }

    #[test]
    fn test_cache_contains_key_and_remove() {
        let cache = FileMetadataCache::new();
        let meta = object_meta("files/test_file", 512);
        let k = meta.location.clone();

        assert!(!cache.contains_key(&k));

        put(&cache, meta, 100);
        assert!(cache.contains_key(&k));
        assert_eq!(cache.len(), 1);

        assert!(cache.remove(&k).is_some());
        assert!(!cache.contains_key(&k));
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.memory_size(), 0);
    }

    #[test]
    fn test_list_entries() {
        use datafusion::execution::cache::cache_manager::FileMetadataCache as FmcTrait;

        let cache = FileMetadataCache::new();
        assert!(FmcTrait::list_entries(&cache).is_empty());

        let meta = object_meta("files/list_test", 256);
        let k = meta.location.clone();
        put(&cache, meta, 100);

        let entries = FmcTrait::list_entries(&cache);
        assert_eq!(entries.len(), 1);
        assert!(entries.contains_key(&k));
    }

    #[test]
    fn test_cache_name() {
        assert_eq!(FileMetadataCache::new().name(), "FileMetadataCache");
    }

    #[test]
    fn test_cache_remove_nonexistent_returns_none() {
        let cache = FileMetadataCache::new();
        assert!(cache.remove(&Path::from("does_not_exist")).is_none());
    }
}
