// src/compute_manager/operators_v2/gpu_v2/mod.rs
//
// Подмодули GPU-оператора v2.
//
//   * queue_v2 — выделенный GPU-тред (стек 32 МБ) + очередь заданий.

#![allow(dead_code, unused_imports)]

pub mod queue_v2;