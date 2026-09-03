// src/layers/ind_rnn/mod.rs

pub mod ind_rnn;

pub mod gpu;   // публичный, но реализация вызывает panic

mod cpu;

pub use ind_rnn::IndRNN;