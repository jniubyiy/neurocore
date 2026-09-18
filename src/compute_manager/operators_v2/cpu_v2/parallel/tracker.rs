// src/compute_manager/cpu/parallel/tracker.rs
//
// Отслеживание состояния чанков во время параллельного прохода.
// Используется для отладки: при включённом PARALLEL_DEBUG
// (см. shared.rs) трекер позволяет увидеть, какие чанки
// выполняются какими воркерами и сколько они занимают времени.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkStatus {
    Pending,
    Assigned,
    InProgress,
    Done,
}

struct ChunkState {
    chunk_id: usize,
    start: usize,
    end: usize,
    status: ChunkStatus,
    assigned_worker: Option<usize>,
    physical_worker: Option<usize>,
    started_at: Option<Instant>,
    finished_at: Option<Instant>,
    duration_ns: u64,
}

#[allow(dead_code)]
pub(super) struct ChunkTracker {
    chunks: Vec<ChunkState>,
    created_at: Instant,
    last_change: Instant,
    total_changes: usize,
}

impl ChunkTracker {
    pub fn new(chunks: &[(usize, usize, usize)]) -> Self {
        let now = Instant::now();
        let entries = chunks
            .iter()
            .enumerate()
            .map(|(chunk_id, &(start, _size, end))| ChunkState {
                chunk_id,
                start,
                end,
                status: ChunkStatus::Pending,
                assigned_worker: None,
                physical_worker: None,
                started_at: None,
                finished_at: None,
                duration_ns: 0,
            })
            .collect();
        Self {
            chunks: entries,
            created_at: now,
            last_change: now,
            total_changes: 0,
        }
    }

    pub fn mark_assigned(&mut self, chunk_id: usize, logical_worker_id: usize) {
        if let Some(c) = self.chunks.get_mut(chunk_id) {
            c.status = ChunkStatus::Assigned;
            c.assigned_worker = Some(logical_worker_id);
            self.last_change = Instant::now();
            self.total_changes += 1;
        }
    }

    pub fn mark_in_progress(&mut self, chunk_id: usize, physical_worker_id: usize) {
        if let Some(c) = self.chunks.get_mut(chunk_id) {
            c.status = ChunkStatus::InProgress;
            c.physical_worker = Some(physical_worker_id);
            c.started_at = Some(Instant::now());
            self.last_change = Instant::now();
            self.total_changes += 1;
        }
    }

    pub fn mark_done(&mut self, chunk_id: usize, duration_ns: u64) {
        if let Some(c) = self.chunks.get_mut(chunk_id) {
            c.status = ChunkStatus::Done;
            c.finished_at = Some(Instant::now());
            c.duration_ns = duration_ns;
            self.last_change = Instant::now();
            self.total_changes += 1;
        }
    }

    #[allow(dead_code)]
    pub fn total_changes(&self) -> usize {
        self.total_changes
    }

    #[allow(dead_code)]
    pub fn dump(&self) -> String {
        let elapsed = self.created_at.elapsed();
        let mut s = String::new();

        let n_total = self.chunks.len();
        let n_done = self.chunks.iter().filter(|c| c.status == ChunkStatus::Done).count();
        let n_in_progress = self
            .chunks
            .iter()
            .filter(|c| c.status == ChunkStatus::InProgress)
            .count();
        let n_assigned = self
            .chunks
            .iter()
            .filter(|c| c.status == ChunkStatus::Assigned)
            .count();
        let n_pending = self
            .chunks
            .iter()
            .filter(|c| c.status == ChunkStatus::Pending)
            .count();

        let _ = writeln!(s, "=== ChunkTracker dump @ T+{:.3}s ===", elapsed.as_secs_f64());
        let _ = writeln!(
            s,
            "total={} | done={} | in_progress={} | assigned={} | pending={}",
            n_total, n_done, n_in_progress, n_assigned, n_pending
        );
        let _ = writeln!(
            s,
            "last_change: T+{:.3}s ({} changes total)",
            self.last_change.duration_since(self.created_at).as_secs_f64(),
            self.total_changes
        );

        let _ = writeln!(s, "--- done ---");
        let mut any = false;
        for c in self.chunks.iter().filter(|c| c.status == ChunkStatus::Done) {
            any = true;
            let _ = writeln!(
                s,
                "  chunk {:>3}  range [{:>3}..{:<3})  assigned={:?}  physical={:?}  dur={:.3}us",
                c.chunk_id, c.start, c.end, c.assigned_worker, c.physical_worker,
                c.duration_ns as f64 / 1000.0
            );
        }
        if !any { let _ = writeln!(s, "  (none)"); }

        let _ = writeln!(s, "--- in_progress ---");
        any = false;
        for c in self.chunks.iter().filter(|c| c.status == ChunkStatus::InProgress) {
            any = true;
            let running = c.started_at
                .map(|a| format!("{:.3}s", a.elapsed().as_secs_f64()))
                .unwrap_or_else(|| "?".into());
            let _ = writeln!(
                s,
                "  chunk {:>3}  range [{:>3}..{:<3})  assigned={:?}  physical={:?}  running={}",
                c.chunk_id, c.start, c.end, c.assigned_worker, c.physical_worker, running
            );
        }
        if !any { let _ = writeln!(s, "  (none)"); }

        let _ = writeln!(s, "--- assigned (not started) ---");
        any = false;
        for c in self.chunks.iter().filter(|c| c.status == ChunkStatus::Assigned) {
            any = true;
            let _ = writeln!(
                s,
                "  chunk {:>3}  range [{:>3}..{:<3})  assigned_worker={:?}",
                c.chunk_id, c.start, c.end, c.assigned_worker
            );
        }
        if !any { let _ = writeln!(s, "  (none)"); }

        let _ = writeln!(s, "--- pending ---");
        any = false;
        for c in self.chunks.iter().filter(|c| c.status == ChunkStatus::Pending) {
            any = true;
            let _ = writeln!(s, "  chunk {:>3}  range [{:>3}..{:<3})", c.chunk_id, c.start, c.end);
        }
        if !any { let _ = writeln!(s, "  (none)"); }

        let _ = writeln!(s, "--- logical worker stats ---");
        let mut lmap: BTreeMap<usize, (usize, usize, usize, Vec<usize>)> = BTreeMap::new();
        for c in &self.chunks {
            if let Some(w) = c.assigned_worker {
                let e = lmap.entry(w).or_insert((0, 0, 0, Vec::new()));
                match c.status {
                    ChunkStatus::Done => e.0 += 1,
                    ChunkStatus::InProgress => e.1 += 1,
                    ChunkStatus::Assigned => e.2 += 1,
                    ChunkStatus::Pending => {}
                }
                e.3.push(c.chunk_id);
            }
        }
        if lmap.is_empty() {
            let _ = writeln!(s, "  (none)");
        } else {
            for (w, (d, ip, a, ids)) in lmap {
                let _ = writeln!(
                    s,
                    "  logical worker {}: done={}, in_progress={}, assigned={}, chunks={:?}",
                    w, d, ip, a, ids
                );
            }
        }

        let _ = writeln!(s, "--- physical worker stats ---");
        let mut pmap: BTreeMap<usize, (usize, usize, Vec<usize>)> = BTreeMap::new();
        for c in &self.chunks {
            if let Some(w) = c.physical_worker {
                let e = pmap.entry(w).or_insert((0, 0, Vec::new()));
                match c.status {
                    ChunkStatus::Done => e.0 += 1,
                    ChunkStatus::InProgress => e.1 += 1,
                    _ => {}
                }
                e.2.push(c.chunk_id);
            }
        }
        if pmap.is_empty() {
            let _ = writeln!(s, "  (no physical worker has touched any chunk)");
        } else {
            for (w, (d, ip, ids)) in pmap {
                let _ = writeln!(
                    s,
                    "  physical worker {}: done={}, in_progress={}, chunks={:?}",
                    w, d, ip, ids
                );
            }
        }

        s
    }
}