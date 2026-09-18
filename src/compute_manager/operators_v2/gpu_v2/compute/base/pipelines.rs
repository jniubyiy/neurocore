// src/compute_manager/gpu/compute/base/pipelines.rs
//
//! Ленивый доступ к пайплайнам слоёв, оптимизаторов и функций потерь.
//!
//! Каждый метод инициализирует соответствующий `OnceLock` при первом
//! обращении. Создание пайплайна происходит на устройстве `self.context.device`.

use super::GpuCompute;

use crate::layers::relu::gpu::pipeline::ReLUPipelines;
use crate::layers::sigmoid::gpu::pipeline::SigmoidPipelines;
use crate::layers::tanh::gpu::pipeline::TanhPipelines;
use crate::layers::leaky_relu::gpu::pipeline::LeakyReLUPipelines;
use crate::layers::linear::gpu::pipeline::LinearPipelines;
use crate::layers::soft_sparse_gate::gpu::pipeline::SoftSparseGatePipelines;
use crate::layers::soft_keep_gate::gpu::pipeline::SoftKeepGatePipelines;
use crate::layers::dual_anchor::gpu::pipeline::DualAnchorPipelines;
use crate::layers::softmax::gpu::pipeline::SoftmaxPipelines;
use crate::layers::memory::gpu::pipeline::MemoryPipelines;
use crate::layers::splitter::gpu::pipeline::SplitterPipelines;
use crate::layers::combiner::gpu::pipeline::CombinerPipelines;
use crate::layers::adaptive_activation::gpu::pipeline::AdaptivePerFeatureActivationPipelines;

use crate::layers::dual_slope_relu::gpu::pipeline::DualSlopeReLUPipelines;
use crate::layers::learnable_mish::gpu::pipeline::LearnableMishPipelines;
use crate::layers::learnable_softplus::gpu::pipeline::LearnableSoftplusPipelines;
use crate::layers::rms_norm_learnable_eps::gpu::pipeline::RMSNormWithLearnableEpsilonPipelines;
use crate::layers::adaptive_dropout::gpu::pipeline::AdaptiveDropoutPipelines;
use crate::layers::feature_fusion::gpu::pipeline::FeatureFusionPipelines;
use crate::layers::sparse_feature_selection_gate::gpu::pipeline::SparseFeatureSelectionGatePipelines;
use crate::layers::multi_resolution_kan_linear::gpu::pipeline::MultiResolutionKANLinearPipelines;
use crate::layers::adaptive_normalization::gpu::pipeline::AdaptiveNormalizationPipelines;
use crate::layers::batch_renorm::gpu::pipeline::BatchRenormPipelines;
use crate::layers::concrete_dropout::gpu::pipeline::ConcreteDropoutPipelines;
use crate::layers::mamba::gpu::pipeline::MambaPipelines;
use crate::layers::ind_rnn::gpu::pipeline::IndRNNPipelines;
use crate::layers::spectral_norm_linear::gpu::pipeline::SpectrallyNormalizedLinearPipelines;

use crate::layers::linear_attention::gpu::pipeline::LinearAttentionPipelines;
use crate::layers::relative_position_attention::gpu::pipeline::RelativePositionAttentionPipelines;
use crate::layers::per_feature_attention::gpu::pipeline::PerFeatureAttentionPipelines;

use crate::optimizers::scale_gradient::gpu::pipeline::ScaleGradientPipelines;
use crate::optimizers::add_weight_decay::gpu::pipeline::AddWeightDecayPipelines;
use crate::optimizers::gradient_clip::gpu::pipeline::GradientClipPipelines;
use crate::optimizers::momentum::gpu::pipeline::MomentumPipelines;
use crate::optimizers::nesterov_momentum::gpu::pipeline::NesterovMomentumPipelines;
use crate::optimizers::adam::gpu::pipeline::AdamPipelines;
use crate::optimizers::apply_update::gpu::pipeline::ApplyUpdatePipelines;

use crate::losses::sub::gpu::pipeline::SubPipelines;
use crate::losses::square::gpu::pipeline::SquarePipelines;
use crate::losses::abs::gpu::pipeline::AbsPipelines;
use crate::losses::log1p::gpu::pipeline::Log1pPipelines;
use crate::losses::abs_diff::gpu::pipeline::AbsDiffPipelines;
use crate::losses::log::gpu::pipeline::LogPipelines;
use crate::losses::neg::gpu::pipeline::NegPipelines;
use crate::losses::mul::gpu::pipeline::MulPipelines;
use crate::losses::add_scalar::gpu::pipeline::AddScalarPipelines;
use crate::losses::cross_entropy::gpu::pipeline::CrossEntropyPipelines;
use crate::losses::sum_columns::gpu::pipeline::SumColumnsPipelines;

impl GpuCompute {
    // ================ Методы доступа к пайплайнам слоёв ================

    pub fn relu_pipelines(&self) -> &ReLUPipelines {
        self.relu_pipelines
            .get_or_init(|| ReLUPipelines::new(self.context.device.clone()))
    }

    pub fn sigmoid_pipelines(&self) -> &SigmoidPipelines {
        self.sigmoid_pipelines
            .get_or_init(|| SigmoidPipelines::new(self.context.device.clone()))
    }

    pub fn tanh_pipelines(&self) -> &TanhPipelines {
        self.tanh_pipelines
            .get_or_init(|| TanhPipelines::new(self.context.device.clone()))
    }

    pub fn leaky_relu_pipelines(&self) -> &LeakyReLUPipelines {
        self.leaky_relu_pipelines
            .get_or_init(|| LeakyReLUPipelines::new(self.context.device.clone()))
    }

    pub fn linear_pipelines(&self) -> &LinearPipelines {
        self.linear_pipelines
            .get_or_init(|| LinearPipelines::new(self.context.device.clone()))
    }

    pub fn soft_sparse_gate_pipelines(&self) -> &SoftSparseGatePipelines {
        self.soft_sparse_gate_pipelines
            .get_or_init(|| SoftSparseGatePipelines::new(self.context.device.clone()))
    }

    pub fn soft_keep_gate_pipelines(&self) -> &SoftKeepGatePipelines {
        self.soft_keep_gate_pipelines
            .get_or_init(|| SoftKeepGatePipelines::new(self.context.device.clone()))
    }

    pub fn dual_anchor_pipelines(&self) -> &DualAnchorPipelines {
        self.dual_anchor_pipelines
            .get_or_init(|| DualAnchorPipelines::new(self.context.device.clone()))
    }

    pub fn adaptive_activation_pipelines(&self) -> &AdaptivePerFeatureActivationPipelines {
        self.adaptive_activation_pipelines
            .get_or_init(|| AdaptivePerFeatureActivationPipelines::new(self.context.device.clone()))
    }

    pub fn softmax_pipelines(&self) -> &SoftmaxPipelines {
        self.softmax_pipelines
            .get_or_init(|| SoftmaxPipelines::new(self.context.device.clone()))
    }

    pub fn memory_pipelines(&self) -> &MemoryPipelines {
        self.memory_pipelines
            .get_or_init(|| MemoryPipelines::new(self.context.device.clone()))
    }

    pub fn splitter_pipelines(&self) -> &SplitterPipelines {
        self.splitter_pipelines
            .get_or_init(|| SplitterPipelines::new(self.context.device.clone()))
    }

    pub fn combiner_pipelines(&self) -> &CombinerPipelines {
        self.combiner_pipelines
            .get_or_init(|| CombinerPipelines::new(self.context.device.clone()))
    }

    // ================ Методы доступа к пайплайнам новых слоёв ================

    pub fn dual_slope_relu_pipelines(&self) -> &DualSlopeReLUPipelines {
        self.dual_slope_relu_pipelines
            .get_or_init(|| DualSlopeReLUPipelines::new(self.context.device.clone()))
    }

    pub fn learnable_mish_pipelines(&self) -> &LearnableMishPipelines {
        self.learnable_mish_pipelines
            .get_or_init(|| LearnableMishPipelines::new(self.context.device.clone()))
    }

    pub fn learnable_softplus_pipelines(&self) -> &LearnableSoftplusPipelines {
        self.learnable_softplus_pipelines
            .get_or_init(|| LearnableSoftplusPipelines::new(self.context.device.clone()))
    }

    pub fn rms_norm_learnable_eps_pipelines(&self) -> &RMSNormWithLearnableEpsilonPipelines {
        self.rms_norm_learnable_eps_pipelines
            .get_or_init(|| RMSNormWithLearnableEpsilonPipelines::new(self.context.device.clone()))
    }

    pub fn adaptive_dropout_pipelines(&self) -> &AdaptiveDropoutPipelines {
        self.adaptive_dropout_pipelines
            .get_or_init(|| AdaptiveDropoutPipelines::new(self.context.device.clone()))
    }

    pub fn feature_fusion_pipelines(&self) -> &FeatureFusionPipelines {
        self.feature_fusion_pipelines
            .get_or_init(|| FeatureFusionPipelines::new(self.context.device.clone()))
    }

    pub fn sparse_feature_selection_gate_pipelines(&self) -> &SparseFeatureSelectionGatePipelines {
        self.sparse_feature_selection_gate_pipelines
            .get_or_init(|| SparseFeatureSelectionGatePipelines::new(self.context.device.clone()))
    }

    pub fn multi_resolution_kan_linear_pipelines(&self) -> &MultiResolutionKANLinearPipelines {
        self.multi_resolution_kan_linear_pipelines
            .get_or_init(|| MultiResolutionKANLinearPipelines::new(self.context.device.clone()))
    }

    pub fn adaptive_normalization_pipelines(&self) -> &AdaptiveNormalizationPipelines {
        self.adaptive_normalization_pipelines
            .get_or_init(|| AdaptiveNormalizationPipelines::new(self.context.device.clone()))
    }

    pub fn batch_renorm_pipelines(&self) -> &BatchRenormPipelines {
        self.batch_renorm_pipelines
            .get_or_init(|| BatchRenormPipelines::new(self.context.device.clone()))
    }

    pub fn concrete_dropout_pipelines(&self) -> &ConcreteDropoutPipelines {
        self.concrete_dropout_pipelines
            .get_or_init(|| ConcreteDropoutPipelines::new(self.context.device.clone()))
    }

    pub fn mamba_pipelines(&self) -> &MambaPipelines {
        self.mamba_pipelines
            .get_or_init(|| MambaPipelines::new(self.context.device.clone()))
    }

    pub fn ind_rnn_pipelines(&self) -> &IndRNNPipelines {
        self.ind_rnn_pipelines
            .get_or_init(|| IndRNNPipelines::new(self.context.device.clone()))
    }

    pub fn spectral_norm_linear_pipelines(&self) -> &SpectrallyNormalizedLinearPipelines {
        self.spectral_norm_linear_pipelines
            .get_or_init(|| SpectrallyNormalizedLinearPipelines::new(self.context.device.clone()))
    }

    pub fn linear_attention_pipelines(&self) -> &LinearAttentionPipelines {
        self.linear_attention_pipelines
            .get_or_init(|| LinearAttentionPipelines::new(self.context.device.clone()))
    }

    pub fn relative_position_attention_pipelines(&self) -> &RelativePositionAttentionPipelines {
        self.relative_position_attention_pipelines
            .get_or_init(|| RelativePositionAttentionPipelines::new(self.context.device.clone()))
    }

    pub fn per_feature_attention_pipelines(&self) -> &PerFeatureAttentionPipelines {
        self.per_feature_attention_pipelines
            .get_or_init(|| PerFeatureAttentionPipelines::new(self.context.device.clone()))
    }

    // ================ Методы доступа к пайплайнам оптимизаторов ================

    pub fn scale_gradient_pipelines(&self) -> &ScaleGradientPipelines {
        self.scale_gradient_pipelines
            .get_or_init(|| ScaleGradientPipelines::new(self.context.device.clone()))
    }

    pub fn add_weight_decay_pipelines(&self) -> &AddWeightDecayPipelines {
        self.add_weight_decay_pipelines
            .get_or_init(|| AddWeightDecayPipelines::new(self.context.device.clone()))
    }

    pub fn gradient_clip_pipelines(&self) -> &GradientClipPipelines {
        self.gradient_clip_pipelines
            .get_or_init(|| GradientClipPipelines::new(self.context.device.clone()))
    }

    pub fn momentum_pipelines(&self) -> &MomentumPipelines {
        self.momentum_pipelines
            .get_or_init(|| MomentumPipelines::new(self.context.device.clone()))
    }

    pub fn nesterov_momentum_pipelines(&self) -> &NesterovMomentumPipelines {
        self.nesterov_momentum_pipelines
            .get_or_init(|| NesterovMomentumPipelines::new(self.context.device.clone()))
    }

    pub fn adam_pipelines(&self) -> &AdamPipelines {
        self.adam_pipelines
            .get_or_init(|| AdamPipelines::new(self.context.device.clone()))
    }

    pub fn apply_update_pipelines(&self) -> &ApplyUpdatePipelines {
        self.apply_update_pipelines
            .get_or_init(|| ApplyUpdatePipelines::new(self.context.device.clone()))
    }

    // ================ Методы доступа к пайплайнам функций потерь ================

    pub fn sub_pipelines(&self) -> &SubPipelines {
        self.sub_pipelines
            .get_or_init(|| SubPipelines::new(self.context.device.clone()))
    }

    pub fn square_pipelines(&self) -> &SquarePipelines {
        self.square_pipelines
            .get_or_init(|| SquarePipelines::new(self.context.device.clone()))
    }

    pub fn abs_pipelines(&self) -> &AbsPipelines {
        self.abs_pipelines
            .get_or_init(|| AbsPipelines::new(self.context.device.clone()))
    }

    pub fn log1p_pipelines(&self) -> &Log1pPipelines {
        self.log1p_pipelines
            .get_or_init(|| Log1pPipelines::new(self.context.device.clone()))
    }

    pub fn abs_diff_pipelines(&self) -> &AbsDiffPipelines {
        self.abs_diff_pipelines
            .get_or_init(|| AbsDiffPipelines::new(self.context.device.clone()))
    }

    pub fn log_pipelines(&self) -> &LogPipelines {
        self.log_pipelines
            .get_or_init(|| LogPipelines::new(self.context.device.clone()))
    }

    pub fn neg_pipelines(&self) -> &NegPipelines {
        self.neg_pipelines
            .get_or_init(|| NegPipelines::new(self.context.device.clone()))
    }

    pub fn mul_pipelines(&self) -> &MulPipelines {
        self.mul_pipelines
            .get_or_init(|| MulPipelines::new(self.context.device.clone()))
    }

    pub fn add_scalar_pipelines(&self) -> &AddScalarPipelines {
        self.add_scalar_pipelines
            .get_or_init(|| AddScalarPipelines::new(self.context.device.clone()))
    }

    pub fn cross_entropy_pipelines(&self) -> &CrossEntropyPipelines {
        self.cross_entropy_pipelines
            .get_or_init(|| CrossEntropyPipelines::new(self.context.device.clone()))
    }

    pub fn sum_columns_pipelines(&self) -> &SumColumnsPipelines {
        self.sum_columns_pipelines
            .get_or_init(|| SumColumnsPipelines::new(self.context.device.clone()))
    }
}