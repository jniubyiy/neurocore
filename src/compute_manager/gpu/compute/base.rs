// src/compute_manager/gpu/compute/base.rs

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{
    allocator::StandardCommandBufferAllocator,
    AutoCommandBufferBuilder, BufferCopy, CommandBufferUsage, CopyBufferInfo,
};
use vulkano::descriptor_set::{
    allocator::StandardDescriptorSetAllocator,
    DescriptorSet, WriteDescriptorSet,
};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};

use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::memory_executor::{
    MemoryExecutor,
    types::MemoryDeviceKind,
    executor::RawBufferId,
    BufferPriority,
};
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::compute_manager::memory_executor::matrix_entry::MatrixStorage;

use super::super::init::GpuContext;
use super::super::pipeline::PipelineCache;

// Импорты пайплайнов слоёв
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

// Новые слои
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

// ВНИМАНИЕ: добавлены импорты для LinearAttention и RelativePositionAttention
use crate::layers::linear_attention::gpu::pipeline::LinearAttentionPipelines;
use crate::layers::relative_position_attention::gpu::pipeline::RelativePositionAttentionPipelines;

// Пайплайны оптимизаторов
use crate::optimizers::scale_gradient::gpu::pipeline::ScaleGradientPipelines;
use crate::optimizers::add_weight_decay::gpu::pipeline::AddWeightDecayPipelines;
use crate::optimizers::gradient_clip::gpu::pipeline::GradientClipPipelines;
use crate::optimizers::momentum::gpu::pipeline::MomentumPipelines;
use crate::optimizers::nesterov_momentum::gpu::pipeline::NesterovMomentumPipelines;
use crate::optimizers::adam::gpu::pipeline::AdamPipelines;
use crate::optimizers::apply_update::gpu::pipeline::ApplyUpdatePipelines;

// Пайплайны функций потерь
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

pub struct GpuCompute {
    pub context: Arc<GpuContext>,
    pub pipeline_cache: Arc<PipelineCache>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub memory_executor: Arc<RwLock<MemoryExecutor>>,
    pub gpu_device_id: DeviceId,
    /// Хранилище состояний для каждого слоя Memory по индексу (memory_idx).
    pub memory_states: Mutex<HashMap<usize, (Subbuffer<[f32]>, RawBufferId)>>,

    queue_lock: Mutex<()>,

    // Пайплайны слоёв (ленивая инициализация)
    relu_pipelines: OnceLock<ReLUPipelines>,
    sigmoid_pipelines: OnceLock<SigmoidPipelines>,
    tanh_pipelines: OnceLock<TanhPipelines>,
    leaky_relu_pipelines: OnceLock<LeakyReLUPipelines>,
    linear_pipelines: OnceLock<LinearPipelines>,
    soft_sparse_gate_pipelines: OnceLock<SoftSparseGatePipelines>,
    soft_keep_gate_pipelines: OnceLock<SoftKeepGatePipelines>,
    dual_anchor_pipelines: OnceLock<DualAnchorPipelines>,
    adaptive_activation_pipelines: OnceLock<AdaptivePerFeatureActivationPipelines>,
    softmax_pipelines: OnceLock<SoftmaxPipelines>,
    memory_pipelines: OnceLock<MemoryPipelines>,
    splitter_pipelines: OnceLock<SplitterPipelines>,
    combiner_pipelines: OnceLock<CombinerPipelines>,

    // Новые слои
    dual_slope_relu_pipelines: OnceLock<DualSlopeReLUPipelines>,
    learnable_mish_pipelines: OnceLock<LearnableMishPipelines>,
    learnable_softplus_pipelines: OnceLock<LearnableSoftplusPipelines>,
    rms_norm_learnable_eps_pipelines: OnceLock<RMSNormWithLearnableEpsilonPipelines>,
    adaptive_dropout_pipelines: OnceLock<AdaptiveDropoutPipelines>,
    feature_fusion_pipelines: OnceLock<FeatureFusionPipelines>,
    sparse_feature_selection_gate_pipelines: OnceLock<SparseFeatureSelectionGatePipelines>,
    multi_resolution_kan_linear_pipelines: OnceLock<MultiResolutionKANLinearPipelines>,
    adaptive_normalization_pipelines: OnceLock<AdaptiveNormalizationPipelines>,
    batch_renorm_pipelines: OnceLock<BatchRenormPipelines>,
    concrete_dropout_pipelines: OnceLock<ConcreteDropoutPipelines>,
    mamba_pipelines: OnceLock<MambaPipelines>,
    ind_rnn_pipelines: OnceLock<IndRNNPipelines>,
    spectral_norm_linear_pipelines: OnceLock<SpectrallyNormalizedLinearPipelines>,

    // ВНИМАНИЕ: добавлены поля для новых пайплайнов
    linear_attention_pipelines: OnceLock<LinearAttentionPipelines>,
    relative_position_attention_pipelines: OnceLock<RelativePositionAttentionPipelines>,

    // Пайплайны оптимизаторов
    scale_gradient_pipelines: OnceLock<ScaleGradientPipelines>,
    add_weight_decay_pipelines: OnceLock<AddWeightDecayPipelines>,
    gradient_clip_pipelines: OnceLock<GradientClipPipelines>,
    momentum_pipelines: OnceLock<MomentumPipelines>,
    nesterov_momentum_pipelines: OnceLock<NesterovMomentumPipelines>,
    adam_pipelines: OnceLock<AdamPipelines>,
    apply_update_pipelines: OnceLock<ApplyUpdatePipelines>,

    // Пайплайны функций потерь
    sub_pipelines: OnceLock<SubPipelines>,
    square_pipelines: OnceLock<SquarePipelines>,
    abs_pipelines: OnceLock<AbsPipelines>,
    log1p_pipelines: OnceLock<Log1pPipelines>,
    abs_diff_pipelines: OnceLock<AbsDiffPipelines>,
    log_pipelines: OnceLock<LogPipelines>,
    neg_pipelines: OnceLock<NegPipelines>,
    mul_pipelines: OnceLock<MulPipelines>,
    add_scalar_pipelines: OnceLock<AddScalarPipelines>,
    cross_entropy_pipelines: OnceLock<CrossEntropyPipelines>,
    sum_columns_pipelines: OnceLock<SumColumnsPipelines>,
}

impl GpuCompute {
    pub fn new(
        context: Arc<GpuContext>,
        pipeline_cache: Arc<PipelineCache>,
        memory_executor: Arc<RwLock<MemoryExecutor>>,
        gpu_device_id: DeviceId,
    ) -> Self {
        let descriptor_set_allocator = Arc::new(
            StandardDescriptorSetAllocator::new(context.device.clone(), Default::default()),
        );
        let command_buffer_allocator = Arc::new(
            StandardCommandBufferAllocator::new(context.device.clone(), Default::default()),
        );
        Self {
            context,
            pipeline_cache,
            descriptor_set_allocator,
            command_buffer_allocator,
            memory_executor,
            gpu_device_id,
            memory_states: Mutex::new(HashMap::new()),
            queue_lock: Mutex::new(()),

            relu_pipelines: OnceLock::new(),
            sigmoid_pipelines: OnceLock::new(),
            tanh_pipelines: OnceLock::new(),
            leaky_relu_pipelines: OnceLock::new(),
            linear_pipelines: OnceLock::new(),
            soft_sparse_gate_pipelines: OnceLock::new(),
            soft_keep_gate_pipelines: OnceLock::new(),
            dual_anchor_pipelines: OnceLock::new(),
            adaptive_activation_pipelines: OnceLock::new(),
            softmax_pipelines: OnceLock::new(),
            memory_pipelines: OnceLock::new(),
            splitter_pipelines: OnceLock::new(),
            combiner_pipelines: OnceLock::new(),

            dual_slope_relu_pipelines: OnceLock::new(),
            learnable_mish_pipelines: OnceLock::new(),
            learnable_softplus_pipelines: OnceLock::new(),
            rms_norm_learnable_eps_pipelines: OnceLock::new(),
            adaptive_dropout_pipelines: OnceLock::new(),
            feature_fusion_pipelines: OnceLock::new(),
            sparse_feature_selection_gate_pipelines: OnceLock::new(),
            multi_resolution_kan_linear_pipelines: OnceLock::new(),
            adaptive_normalization_pipelines: OnceLock::new(),
            batch_renorm_pipelines: OnceLock::new(),
            concrete_dropout_pipelines: OnceLock::new(),
            mamba_pipelines: OnceLock::new(),
            ind_rnn_pipelines: OnceLock::new(),
            spectral_norm_linear_pipelines: OnceLock::new(),

            // ВНИМАНИЕ: инициализация новых полей
            linear_attention_pipelines: OnceLock::new(),
            relative_position_attention_pipelines: OnceLock::new(),

            scale_gradient_pipelines: OnceLock::new(),
            add_weight_decay_pipelines: OnceLock::new(),
            gradient_clip_pipelines: OnceLock::new(),
            momentum_pipelines: OnceLock::new(),
            nesterov_momentum_pipelines: OnceLock::new(),
            adam_pipelines: OnceLock::new(),
            apply_update_pipelines: OnceLock::new(),

            sub_pipelines: OnceLock::new(),
            square_pipelines: OnceLock::new(),
            abs_pipelines: OnceLock::new(),
            log1p_pipelines: OnceLock::new(),
            abs_diff_pipelines: OnceLock::new(),
            log_pipelines: OnceLock::new(),
            neg_pipelines: OnceLock::new(),
            mul_pipelines: OnceLock::new(),
            add_scalar_pipelines: OnceLock::new(),
            cross_entropy_pipelines: OnceLock::new(),
            sum_columns_pipelines: OnceLock::new(),
        }
    }

    // ================ Методы доступа к пайплайнам слоёв ================

    pub fn relu_pipelines(&self) -> &ReLUPipelines {
        self.relu_pipelines.get_or_init(|| ReLUPipelines::new(self.context.device.clone()))
    }

    pub fn sigmoid_pipelines(&self) -> &SigmoidPipelines {
        self.sigmoid_pipelines.get_or_init(|| SigmoidPipelines::new(self.context.device.clone()))
    }

    pub fn tanh_pipelines(&self) -> &TanhPipelines {
        self.tanh_pipelines.get_or_init(|| TanhPipelines::new(self.context.device.clone()))
    }

    pub fn leaky_relu_pipelines(&self) -> &LeakyReLUPipelines {
        self.leaky_relu_pipelines.get_or_init(|| LeakyReLUPipelines::new(self.context.device.clone()))
    }

    pub fn linear_pipelines(&self) -> &LinearPipelines {
        self.linear_pipelines.get_or_init(|| LinearPipelines::new(self.context.device.clone()))
    }

    pub fn soft_sparse_gate_pipelines(&self) -> &SoftSparseGatePipelines {
        self.soft_sparse_gate_pipelines.get_or_init(|| SoftSparseGatePipelines::new(self.context.device.clone()))
    }

    pub fn soft_keep_gate_pipelines(&self) -> &SoftKeepGatePipelines {
        self.soft_keep_gate_pipelines.get_or_init(|| SoftKeepGatePipelines::new(self.context.device.clone()))
    }

    pub fn dual_anchor_pipelines(&self) -> &DualAnchorPipelines {
        self.dual_anchor_pipelines.get_or_init(|| DualAnchorPipelines::new(self.context.device.clone()))
    }

    pub fn adaptive_activation_pipelines(&self) -> &AdaptivePerFeatureActivationPipelines {
        self.adaptive_activation_pipelines.get_or_init(|| AdaptivePerFeatureActivationPipelines::new(self.context.device.clone()))
    }

    pub fn softmax_pipelines(&self) -> &SoftmaxPipelines {
        self.softmax_pipelines.get_or_init(|| SoftmaxPipelines::new(self.context.device.clone()))
    }

    pub fn memory_pipelines(&self) -> &MemoryPipelines {
        self.memory_pipelines.get_or_init(|| MemoryPipelines::new(self.context.device.clone()))
    }

    pub fn splitter_pipelines(&self) -> &SplitterPipelines {
        self.splitter_pipelines.get_or_init(|| SplitterPipelines::new(self.context.device.clone()))
    }

    pub fn combiner_pipelines(&self) -> &CombinerPipelines {
        self.combiner_pipelines.get_or_init(|| CombinerPipelines::new(self.context.device.clone()))
    }

    // Новые слои
    pub fn dual_slope_relu_pipelines(&self) -> &DualSlopeReLUPipelines {
        self.dual_slope_relu_pipelines.get_or_init(|| DualSlopeReLUPipelines::new(self.context.device.clone()))
    }

    pub fn learnable_mish_pipelines(&self) -> &LearnableMishPipelines {
        self.learnable_mish_pipelines.get_or_init(|| LearnableMishPipelines::new(self.context.device.clone()))
    }

    pub fn learnable_softplus_pipelines(&self) -> &LearnableSoftplusPipelines {
        self.learnable_softplus_pipelines.get_or_init(|| LearnableSoftplusPipelines::new(self.context.device.clone()))
    }

    pub fn rms_norm_learnable_eps_pipelines(&self) -> &RMSNormWithLearnableEpsilonPipelines {
        self.rms_norm_learnable_eps_pipelines.get_or_init(|| RMSNormWithLearnableEpsilonPipelines::new(self.context.device.clone()))
    }

    pub fn adaptive_dropout_pipelines(&self) -> &AdaptiveDropoutPipelines {
        self.adaptive_dropout_pipelines.get_or_init(|| AdaptiveDropoutPipelines::new(self.context.device.clone()))
    }

    pub fn feature_fusion_pipelines(&self) -> &FeatureFusionPipelines {
        self.feature_fusion_pipelines.get_or_init(|| FeatureFusionPipelines::new(self.context.device.clone()))
    }

    pub fn sparse_feature_selection_gate_pipelines(&self) -> &SparseFeatureSelectionGatePipelines {
        self.sparse_feature_selection_gate_pipelines.get_or_init(|| SparseFeatureSelectionGatePipelines::new(self.context.device.clone()))
    }

    pub fn multi_resolution_kan_linear_pipelines(&self) -> &MultiResolutionKANLinearPipelines {
        self.multi_resolution_kan_linear_pipelines.get_or_init(|| MultiResolutionKANLinearPipelines::new(self.context.device.clone()))
    }

    pub fn adaptive_normalization_pipelines(&self) -> &AdaptiveNormalizationPipelines {
        self.adaptive_normalization_pipelines.get_or_init(|| AdaptiveNormalizationPipelines::new(self.context.device.clone()))
    }

    pub fn batch_renorm_pipelines(&self) -> &BatchRenormPipelines {
        self.batch_renorm_pipelines.get_or_init(|| BatchRenormPipelines::new(self.context.device.clone()))
    }

    pub fn concrete_dropout_pipelines(&self) -> &ConcreteDropoutPipelines {
        self.concrete_dropout_pipelines.get_or_init(|| ConcreteDropoutPipelines::new(self.context.device.clone()))
    }

    pub fn mamba_pipelines(&self) -> &MambaPipelines {
        self.mamba_pipelines.get_or_init(|| MambaPipelines::new(self.context.device.clone()))
    }

    pub fn ind_rnn_pipelines(&self) -> &IndRNNPipelines {
        self.ind_rnn_pipelines.get_or_init(|| IndRNNPipelines::new(self.context.device.clone()))
    }

    pub fn spectral_norm_linear_pipelines(&self) -> &SpectrallyNormalizedLinearPipelines {
        self.spectral_norm_linear_pipelines.get_or_init(|| SpectrallyNormalizedLinearPipelines::new(self.context.device.clone()))
    }

    // ВНИМАНИЕ: добавлены методы для новых пайплайнов
    pub fn linear_attention_pipelines(&self) -> &LinearAttentionPipelines {
        self.linear_attention_pipelines.get_or_init(|| LinearAttentionPipelines::new(self.context.device.clone()))
    }

    pub fn relative_position_attention_pipelines(&self) -> &RelativePositionAttentionPipelines {
        self.relative_position_attention_pipelines.get_or_init(|| RelativePositionAttentionPipelines::new(self.context.device.clone()))
    }

    // ================ Методы доступа к пайплайнам оптимизаторов ================

    pub fn scale_gradient_pipelines(&self) -> &ScaleGradientPipelines {
        self.scale_gradient_pipelines.get_or_init(|| ScaleGradientPipelines::new(self.context.device.clone()))
    }

    pub fn add_weight_decay_pipelines(&self) -> &AddWeightDecayPipelines {
        self.add_weight_decay_pipelines.get_or_init(|| AddWeightDecayPipelines::new(self.context.device.clone()))
    }

    pub fn gradient_clip_pipelines(&self) -> &GradientClipPipelines {
        self.gradient_clip_pipelines.get_or_init(|| GradientClipPipelines::new(self.context.device.clone()))
    }

    pub fn momentum_pipelines(&self) -> &MomentumPipelines {
        self.momentum_pipelines.get_or_init(|| MomentumPipelines::new(self.context.device.clone()))
    }

    pub fn nesterov_momentum_pipelines(&self) -> &NesterovMomentumPipelines {
        self.nesterov_momentum_pipelines.get_or_init(|| NesterovMomentumPipelines::new(self.context.device.clone()))
    }

    pub fn adam_pipelines(&self) -> &AdamPipelines {
        self.adam_pipelines.get_or_init(|| AdamPipelines::new(self.context.device.clone()))
    }

    pub fn apply_update_pipelines(&self) -> &ApplyUpdatePipelines {
        self.apply_update_pipelines.get_or_init(|| ApplyUpdatePipelines::new(self.context.device.clone()))
    }

    // ================ Методы доступа к пайплайнам функций потерь ================

    pub fn sub_pipelines(&self) -> &SubPipelines {
        self.sub_pipelines.get_or_init(|| SubPipelines::new(self.context.device.clone()))
    }

    pub fn square_pipelines(&self) -> &SquarePipelines {
        self.square_pipelines.get_or_init(|| SquarePipelines::new(self.context.device.clone()))
    }

    pub fn abs_pipelines(&self) -> &AbsPipelines {
        self.abs_pipelines.get_or_init(|| AbsPipelines::new(self.context.device.clone()))
    }

    pub fn log1p_pipelines(&self) -> &Log1pPipelines {
        self.log1p_pipelines.get_or_init(|| Log1pPipelines::new(self.context.device.clone()))
    }

    pub fn abs_diff_pipelines(&self) -> &AbsDiffPipelines {
        self.abs_diff_pipelines.get_or_init(|| AbsDiffPipelines::new(self.context.device.clone()))
    }

    pub fn log_pipelines(&self) -> &LogPipelines {
        self.log_pipelines.get_or_init(|| LogPipelines::new(self.context.device.clone()))
    }

    pub fn neg_pipelines(&self) -> &NegPipelines {
        self.neg_pipelines.get_or_init(|| NegPipelines::new(self.context.device.clone()))
    }

    pub fn mul_pipelines(&self) -> &MulPipelines {
        self.mul_pipelines.get_or_init(|| MulPipelines::new(self.context.device.clone()))
    }

    pub fn add_scalar_pipelines(&self) -> &AddScalarPipelines {
        self.add_scalar_pipelines.get_or_init(|| AddScalarPipelines::new(self.context.device.clone()))
    }

    pub fn cross_entropy_pipelines(&self) -> &CrossEntropyPipelines {
        self.cross_entropy_pipelines.get_or_init(|| CrossEntropyPipelines::new(self.context.device.clone()))
    }

    pub fn sum_columns_pipelines(&self) -> &SumColumnsPipelines {
        self.sum_columns_pipelines.get_or_init(|| SumColumnsPipelines::new(self.context.device.clone()))
    }

    // --- Временные буферы ---

    pub fn acquire_temp_buffer(
        &self,
        elements: usize,
    ) -> (Subbuffer<[f32]>, RawBufferId) {
        let kind = MemoryDeviceKind::DeviceVram(self.gpu_device_id);
        self.memory_executor.write().unwrap().acquire_temp_buffer(kind, elements)
    }

    pub fn acquire_staging_buffer(
        &self,
        elements: usize,
    ) -> (Subbuffer<[f32]>, RawBufferId) {
        self.memory_executor.write().unwrap().acquire_temp_buffer(MemoryDeviceKind::HostRam, elements)
    }

    pub fn release_temp_buffer(
        &self,
        buffer: Subbuffer<[f32]>,
        raw_id: RawBufferId,
    ) {
        let kind = MemoryDeviceKind::DeviceVram(self.gpu_device_id);
        self.memory_executor.write().unwrap().release_temp_buffer(kind, buffer, raw_id);
    }

    pub fn release_staging_buffer(
        &self,
        buffer: Subbuffer<[f32]>,
        raw_id: RawBufferId,
    ) {
        self.memory_executor.write().unwrap().release_temp_buffer(MemoryDeviceKind::HostRam, buffer, raw_id);
    }

    // --- Загрузка данных ---

    pub fn upload_to_temp_buffer(
        &self,
        data: &[f32],
    ) -> (Subbuffer<[f32]>, RawBufferId) {
        let elements = data.len();
        let (gpu_buf, raw_id) = self.acquire_temp_buffer(elements);

        let (staging_buf, staging_raw) = self.acquire_staging_buffer(elements);
        {
            let mut write_guard = staging_buf.write().expect("write staging buffer");
            write_guard[..elements].copy_from_slice(data);
        }
        self.copy_buffer_sync(staging_buf.clone(), gpu_buf.clone());
        self.release_staging_buffer(staging_buf, staging_raw);

        (gpu_buf, raw_id)
    }

    // --- Копирование между Subbuffer'ами ---

    pub fn copy_buffer_sync(&self, src: Subbuffer<[f32]>, dst: Subbuffer<[f32]>) {
        let _lock = self.queue_lock.lock().unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .copy_buffer(CopyBufferInfo::buffers(src, dst))
            .unwrap();
        let cb = builder.build().unwrap();
        let future = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
    }

    // --- Диспатч ---

    pub fn run_compute_shader<const N: usize>(
        &self,
        pipeline: &Arc<vulkano::pipeline::ComputePipeline>,
        buffers: &[(u32, Subbuffer<[f32]>)],
        push_constants: &[u32; N],
        total_elements: usize,
    ) {
        let dispatch_dim = [((total_elements + 255) / 256) as u32, 1, 1];
        self.run_compute_shader_with_dispatch(pipeline, buffers, push_constants, dispatch_dim);
    }

    pub fn run_compute_shader_with_dispatch<const N: usize>(
        &self,
        pipeline: &Arc<vulkano::pipeline::ComputePipeline>,
        buffers: &[(u32, Subbuffer<[f32]>)],
        push_constants: &[u32; N],
        dispatch_dim: [u32; 3],
    ) {
        let _lock = self.queue_lock.lock().unwrap();

        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let writes: Vec<WriteDescriptorSet> = buffers
            .iter()
            .map(|(binding, buf)| WriteDescriptorSet::buffer(*binding, buf.clone()))
            .collect();

        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            writes,
            [],
        )
        .expect("descriptor set");

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("command buffer builder");

        unsafe {
            builder
                .bind_pipeline_compute(pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(pipeline.layout().clone(), 0, *push_constants)
                .unwrap()
                .dispatch(dispatch_dim)
                .unwrap();
        }

        let command_buffer = builder.build().expect("build command buffer");
        let future = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
    }

    pub fn run_compute_shader_2d<const N: usize>(
        &self,
        pipeline: &Arc<vulkano::pipeline::ComputePipeline>,
        buffers: &[(u32, Subbuffer<[f32]>)],
        push_constants: &[u32; N],
        dispatch_dim: [u32; 3],
    ) {
        self.run_compute_shader_with_dispatch(pipeline, buffers, push_constants, dispatch_dim);
    }

    // ===================================================================
    // МЕТОДЫ ДЛЯ РАБОТЫ С MatrixBufferHandle
    // ===================================================================

    pub fn allocate_gpu_matrix_handle(&self, rows: usize, cols: usize) -> MatrixBufferHandle {
        let mut mem = self.memory_executor.write().unwrap();
        mem.acquire_matrix_handle(
            rows,
            cols,
            MemoryDeviceKind::DeviceVram(self.gpu_device_id),
            BufferPriority::Medium,
        )
        .expect("Failed to allocate GPU MatrixBufferHandle")
    }

    pub fn allocate_cpu_matrix_handle(&self, rows: usize, cols: usize) -> MatrixBufferHandle {
        let mut mem = self.memory_executor.write().unwrap();
        mem.acquire_matrix_handle(
            rows,
            cols,
            MemoryDeviceKind::HostRam,
            BufferPriority::Medium,
        )
        .expect("Failed to allocate CPU MatrixBufferHandle")
    }

    pub fn upload_vec_to_gpu_handle(
        &self,
        data: &[f32],
        rows: usize,
        cols: usize,
    ) -> MatrixBufferHandle {
        assert_eq!(data.len(), rows * cols, "Data length must match matrix size");
        let gpu_handle = self.allocate_gpu_matrix_handle(rows, cols);
        self.copy_slice_to_gpu_handle(&gpu_handle, data);
        gpu_handle
    }

    pub fn copy_slice_to_gpu_handle(&self, handle: &MatrixBufferHandle, data: &[f32]) {
        assert!(handle.is_gpu(), "Handle must be GPU");
        let elements = handle.rows() * handle.cols();
        assert_eq!(data.len(), elements, "Data length must match handle size");

        let gpu_buf = self.get_gpu_subbuffer_from_handle(handle);
        let (staging_buf, staging_raw) = self.acquire_staging_buffer(elements);
        {
            let mut write_guard = staging_buf.write().expect("write staging buffer");
            write_guard[..elements].copy_from_slice(data);
        }
        self.copy_buffer_sync(staging_buf.clone(), gpu_buf);
        self.release_staging_buffer(staging_buf, staging_raw);
    }

    pub fn download_gpu_handle_to_cpu_handle(&self, handle: &MatrixBufferHandle) -> MatrixBufferHandle {
        assert!(handle.is_gpu(), "Handle must be GPU");
        let elements = handle.rows() * handle.cols();

        let gpu_buf = self.get_gpu_subbuffer_from_handle(handle);
        let (staging_buf, staging_raw) = self.acquire_staging_buffer(elements);
        self.copy_buffer_sync(gpu_buf, staging_buf.clone());

        let cpu_handle = self.allocate_cpu_matrix_handle(handle.rows(), handle.cols());

        {
            let staging_guard = staging_buf.read().expect("read staging buffer");
            let staging_slice = &staging_guard[..elements];
            let mut cpu_guard = cpu_handle.write();
            let cpu_slice = cpu_guard.as_slice_mut().expect("CPU handle must be CPU");
            cpu_slice.copy_from_slice(staging_slice);
        }

        self.release_staging_buffer(staging_buf, staging_raw);
        cpu_handle
    }

    pub fn download_gpu_handle_to_vec(&self, handle: &MatrixBufferHandle) -> Vec<f32> {
        assert!(handle.is_gpu(), "Handle must be GPU");
        let elements = handle.rows() * handle.cols();

        let gpu_buf = self.get_gpu_subbuffer_from_handle(handle);
        let (staging_buf, staging_raw) = self.acquire_staging_buffer(elements);
        self.copy_buffer_sync(gpu_buf, staging_buf.clone());

        let data = {
            let staging_guard = staging_buf.read().expect("read staging buffer");
            staging_guard[..elements].to_vec()
        };

        self.release_staging_buffer(staging_buf, staging_raw);
        data
    }

    pub fn fill_gpu_handle(&self, handle: &MatrixBufferHandle, value: f32) {
        let elements = handle.rows() * handle.cols();
        let data = vec![value; elements];
        self.copy_slice_to_gpu_handle(handle, &data);
    }

    pub fn copy_cpu_to_gpu_handle(
        &self,
        src: &MatrixBufferHandle,
        dst: &MatrixBufferHandle,
    ) {
        assert!(!src.is_gpu(), "Source must be CPU");
        assert!(dst.is_gpu(), "Destination must be GPU");
        let elements = src.rows() * src.cols();
        assert_eq!(elements, dst.rows() * dst.cols(), "Buffer sizes must match");

        let src_guard = src.read();
        let src_slice = src_guard.as_slice().expect("Source is not CPU");
        self.copy_slice_to_gpu_handle(dst, src_slice);
    }

    pub fn copy_gpu_to_cpu_handle(
        &self,
        src: &MatrixBufferHandle,
        dst: &MatrixBufferHandle,
    ) {
        assert!(src.is_gpu(), "Source must be GPU");
        assert!(!dst.is_gpu(), "Destination must be CPU");
        let elements = src.rows() * src.cols();
        assert_eq!(elements, dst.rows() * dst.cols(), "Buffer sizes must match");

        let gpu_buf = self.get_gpu_subbuffer_from_handle(src);
        let (staging_buf, staging_raw) = self.acquire_staging_buffer(elements);

        self.copy_buffer_sync(gpu_buf, staging_buf.clone());

        {
            let staging_guard = staging_buf.read().expect("read staging buffer");
            let mut dst_guard = dst.write();
            let dst_slice = dst_guard.as_slice_mut().expect("Destination is not CPU");
            dst_slice.copy_from_slice(&staging_guard[..elements]);
        }

        self.release_staging_buffer(staging_buf, staging_raw);
    }

    pub fn copy_gpu_handle_to_gpu_handle(
        &self,
        src: &MatrixBufferHandle,
        dst: &MatrixBufferHandle,
    ) {
        assert!(src.is_gpu(), "Source must be GPU");
        assert!(dst.is_gpu(), "Destination must be GPU");
        let src_buf = self.get_gpu_subbuffer_from_handle(src);
        let dst_buf = self.get_gpu_subbuffer_from_handle(dst);
        self.copy_buffer_sync(src_buf, dst_buf);
    }

    pub fn copy_gpu_handle_region(
        &self,
        src: &MatrixBufferHandle,
        dst: &MatrixBufferHandle,
        src_offset: usize,
        dst_offset: usize,
        elements: usize,
    ) {
        assert!(src.is_gpu(), "Source must be GPU");
        assert!(dst.is_gpu(), "Destination must be GPU");

        let _lock = self.queue_lock.lock().unwrap();

        let elem_size = std::mem::size_of::<f32>() as u64;
        let src_start_byte = src_offset as u64 * elem_size;
        let dst_start_byte = dst_offset as u64 * elem_size;
        let byte_len = elements as u64 * elem_size;

        let src_full = self.get_gpu_subbuffer_from_handle(src);
        let dst_full = self.get_gpu_subbuffer_from_handle(dst);

        let src_u8 = src_full.into_bytes();
        let dst_u8 = dst_full.into_bytes();

        let region = BufferCopy {
            src_offset: src_start_byte,
            dst_offset: dst_start_byte,
            size: byte_len,
            ..Default::default()
        };

        let mut info = CopyBufferInfo::buffers(src_u8, dst_u8);
        info.regions = vec![region].into();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(info)
            .unwrap();

        let cb = builder.build().unwrap();
        let future = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
    }

    pub(crate) fn get_gpu_subbuffer_from_handle(&self, handle: &MatrixBufferHandle) -> Subbuffer<[f32]> {
        let mem = self.memory_executor.read().unwrap();
        let entry = mem.get_matrix_entry(handle.id())
            .expect("MatrixBufferHandle: entry not found");
        match &entry.storage {
            MatrixStorage::Gpu { buffer, .. } => buffer.clone(),
            _ => panic!("Expected GPU storage for handle"),
        }
    }
}