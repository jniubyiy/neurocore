// src/layers/buffered_context.rs

use crate::compute_manager::operators_v2::memory_v2::buffer::MatrixBufferHandle;

#[derive(Clone)]
pub struct CpuLinearAttentionHead {
    pub q: MatrixBufferHandle,
    pub k: MatrixBufferHandle,
    pub v: MatrixBufferHandle,
    pub kv: MatrixBufferHandle,
    pub z: MatrixBufferHandle,
    pub attn_out: MatrixBufferHandle,
    pub y: MatrixBufferHandle,
    pub weight: f32,
    pub head_index: usize,
}

#[derive(Clone)]
pub struct CpuPerFeatureAttentionHead {
    pub x: MatrixBufferHandle,
    pub dx: MatrixBufferHandle,
    pub q_raw: MatrixBufferHandle,
    pub k_raw: MatrixBufferHandle,
    pub v_raw: MatrixBufferHandle,
    pub q_phi: MatrixBufferHandle,
    pub k_phi: MatrixBufferHandle,
    pub kv: MatrixBufferHandle,
    pub z: MatrixBufferHandle,
    pub attn: MatrixBufferHandle,
    pub self_bias: f32,
    pub head_index: usize,
}

#[derive(Clone)]
pub enum BufferedContext {
    Linear { input: MatrixBufferHandle },
    ReLU { input: MatrixBufferHandle },
    Sigmoid { output: MatrixBufferHandle },
    Tanh { output: MatrixBufferHandle },
    Softmax { output: MatrixBufferHandle },
    Memory {
        input: MatrixBufferHandle,
        min_cells: MatrixBufferHandle,
        max_cells: MatrixBufferHandle,
    },
    LeakyReLU { input: MatrixBufferHandle },
    SoftSparseGate { input: MatrixBufferHandle },
    SoftKeepGate { input: MatrixBufferHandle },
    DualAnchor1D { input: MatrixBufferHandle },
    AdaptiveActivation { input: MatrixBufferHandle },
    AdaptiveNormalization { input: MatrixBufferHandle },
    BatchRenorm {
        input: MatrixBufferHandle,
        mean: Vec<f32>,
        var: Vec<f32>,
        use_batch_stats: bool,
    },
    ConcreteDropout {
        input: MatrixBufferHandle,
        arg: MatrixBufferHandle,
    },
    Mamba {
        input: MatrixBufferHandle,
        h_all: MatrixBufferHandle,
        a_bar: MatrixBufferHandle,
        b_bar: MatrixBufferHandle,
    },
    LinearAttention {
        input: MatrixBufferHandle,
        cpu_heads: Vec<CpuLinearAttentionHead>,
        h_raw: f32,
        h_soft: f32,
        batch: usize,
        seq: usize,
        d_model: usize,
        d_head: usize,
        q_raw: Option<MatrixBufferHandle>,
        k_raw: Option<MatrixBufferHandle>,
        v_raw: Option<MatrixBufferHandle>,
        q_phi: Option<MatrixBufferHandle>,
        k_phi: Option<MatrixBufferHandle>,
        kv: Option<MatrixBufferHandle>,
        z: Option<MatrixBufferHandle>,
    },
    RelativePositionAttention {
        input: MatrixBufferHandle,
        q: Option<MatrixBufferHandle>,
        k: Option<MatrixBufferHandle>,
        v: Option<MatrixBufferHandle>,
        scores: Option<MatrixBufferHandle>,
        weights: Option<MatrixBufferHandle>,
        attn_out: Option<MatrixBufferHandle>,
    },
    IndRNN {
        input: MatrixBufferHandle,
        h_all: MatrixBufferHandle,
    },
    SpectralNormLinear {
        input: MatrixBufferHandle,
        u_state: MatrixBufferHandle,
        v_state: MatrixBufferHandle,
        sigma_state: MatrixBufferHandle,
    },
    Identity { input: MatrixBufferHandle },
    SplitterConnector { input: MatrixBufferHandle },
    CombinerConnector { inputs: Vec<MatrixBufferHandle> },
    Splitter {
        input: MatrixBufferHandle,
        pre_a: MatrixBufferHandle,
        pre_b: MatrixBufferHandle,
    },
    Combiner {
        input_a: MatrixBufferHandle,
        input_b: MatrixBufferHandle,
        pre_act: MatrixBufferHandle,
    },

    // ================= Новые слои =================
    DualSlopeReLU { input: MatrixBufferHandle },
    LearnableMish { input: MatrixBufferHandle },
    LearnableSoftplus { input: MatrixBufferHandle },
    RMSNormWithLearnableEpsilon { input: MatrixBufferHandle },
    AdaptiveDropout {
        input: MatrixBufferHandle,
        mask: MatrixBufferHandle,
        arg: MatrixBufferHandle,
    },
    FeatureFusion { input: MatrixBufferHandle },
    SparseFeatureSelectionGate { input: MatrixBufferHandle },
    MultiResolutionKANLinear { input: MatrixBufferHandle },

    // ================= PerFeatureAttention =================
    /// CPU-путь. Все state-буферы живут внутри `cpu_heads`.
    PerFeatureAttention {
        input: MatrixBufferHandle,
        cpu_heads: Vec<CpuPerFeatureAttentionHead>,
        batch: usize,
        seq_len: usize,
        d_model: usize,
        d_head: usize,
    },

    /// GPU-путь. Все state-буферы — отдельные GPU-дескрипторы.
    PerFeatureAttentionGpu {
        input: MatrixBufferHandle,
        q_raw: MatrixBufferHandle,
        k_raw: MatrixBufferHandle,
        v_raw: MatrixBufferHandle,
        q_phi: MatrixBufferHandle,
        k_phi: MatrixBufferHandle,
        kv: MatrixBufferHandle,
        z: MatrixBufferHandle,
        attn: MatrixBufferHandle,
        denom: MatrixBufferHandle,
        batch: usize,
        seq_len: usize,
        d_model: usize,
        d_head: usize,
    },
}