// src/training_plan/profiling.rs

use std::collections::HashMap;
use std::time::Instant;

/// Запись о времени выполнения одного сегмента или операции.
#[derive(Debug, Clone)]
pub struct TimingRecord {
    pub segment_index: usize,
    pub layer_name: String,
    pub device: String,
    pub phase: String,
    pub duration_ns: u64,
}

/// Запись об использовании памяти до и после операции.
#[derive(Debug, Clone)]
pub struct MemoryRecord {
    pub segment_index: usize,
    pub device: String,
    pub phase: String,
    pub before_bytes: usize,
    pub after_bytes: usize,
    pub delta_bytes: i64,
}

/// Режим профилирования.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProfileMode {
    None,
    Time,
    Memory,
    Full,
}

/// Сборщик метрик производительности.
pub struct Profiler {
    pub mode: ProfileMode,
    pub timings: Vec<TimingRecord>,
    pub memory_records: Vec<MemoryRecord>,
    start_time: Instant,
}

impl Profiler {
    pub fn new(mode: ProfileMode) -> Self {
        Profiler {
            mode,
            timings: Vec::new(),
            memory_records: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Зафиксировать время выполнения сегмента.
    pub fn record_timing(
        &mut self,
        segment_index: usize,
        layer_name: &str,
        device: &str,
        phase: &str,
        duration_ns: u64,
    ) {
        if self.mode == ProfileMode::Time || self.mode == ProfileMode::Full {
            self.timings.push(TimingRecord {
                segment_index,
                layer_name: layer_name.to_string(),
                device: device.to_string(),
                phase: phase.to_string(),
                duration_ns,
            });
        }
    }

    /// Зафиксировать изменение памяти.
    pub fn record_memory(
        &mut self,
        segment_index: usize,
        device: &str,
        phase: &str,
        before_bytes: usize,
        after_bytes: usize,
    ) {
        if self.mode == ProfileMode::Memory || self.mode == ProfileMode::Full {
            let delta = after_bytes as i64 - before_bytes as i64;
            self.memory_records.push(MemoryRecord {
                segment_index,
                device: device.to_string(),
                phase: phase.to_string(),
                before_bytes,
                after_bytes,
                delta_bytes: delta,
            });
        }
    }

    /// Получить итоговый результат профилирования.
    pub fn finish(self) -> ProfileResult {
        let total_time = self.start_time.elapsed().as_secs_f64();
        let mut time_by_device: HashMap<String, f64> = HashMap::new();
        let mut memory_peak: HashMap<String, usize> = HashMap::new();

        for rec in &self.timings {
            *time_by_device.entry(rec.device.clone()).or_insert(0.0) +=
                rec.duration_ns as f64 / 1_000_000_000.0;
        }

        for rec in &self.memory_records {
            let peak = memory_peak.entry(rec.device.clone()).or_insert(0);
            if rec.after_bytes > *peak {
                *peak = rec.after_bytes;
            }
        }

        ProfileResult {
            total_time_secs: total_time,
            timings: self.timings,
            memory_records: self.memory_records,
            time_by_device,
            memory_peak_bytes_by_device: memory_peak,
        }
    }
}

/// Итоговый отчёт профилирования.
#[derive(Debug, Clone)]
pub struct ProfileResult {
    pub total_time_secs: f64,
    pub timings: Vec<TimingRecord>,
    pub memory_records: Vec<MemoryRecord>,
    pub time_by_device: HashMap<String, f64>,
    pub memory_peak_bytes_by_device: HashMap<String, usize>,
}

impl ProfileResult {
    /// Сформировать текстовый отчёт.
    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Profile Report ===\n");
        out.push_str(&format!("Total wall time: {:.4} s\n", self.total_time_secs));
        out.push_str("\n--- Time by Device ---\n");
        for (dev, secs) in &self.time_by_device {
            out.push_str(&format!("  {}: {:.4} s\n", dev, secs));
        }
        out.push_str("\n--- Memory Peak by Device ---\n");
        for (dev, bytes) in &self.memory_peak_bytes_by_device {
            out.push_str(&format!(
                "  {}: {:.2} MB\n",
                dev,
                *bytes as f64 / 1_048_576.0
            ));
        }
        out.push_str("\n--- Detailed Timings ---\n");
        for t in &self.timings {
            out.push_str(&format!(
                "  [{}] {} {} {}: {:.3} ms\n",
                t.segment_index,
                t.layer_name,
                t.device,
                t.phase,
                t.duration_ns as f64 / 1_000_000.0
            ));
        }
        out.push_str("\n--- Detailed Memory Deltas ---\n");
        for m in &self.memory_records {
            let delta_mb = m.delta_bytes as f64 / 1_048_576.0;
            out.push_str(&format!(
                "  [{}] {} {} {} -> {} (delta {:.2} MB)\n",
                m.segment_index, m.device, m.phase, m.before_bytes, m.after_bytes, delta_mb
            ));
        }
        out
    }
}