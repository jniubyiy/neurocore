// src/compute_manager/gpu/compute/base.rs

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

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

pub struct GpuCompute {
    pub context: Arc<GpuContext>,
    pub pipeline_cache: Arc<PipelineCache>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub memory_executor: Arc<Mutex<MemoryExecutor>>,
    pub gpu_device_id: DeviceId,
    /// Хранилище состояний для каждого слоя Memory по индексу (memory_idx).
    pub memory_states: Mutex<HashMap<usize, (Subbuffer<[f32]>, RawBufferId)>>,
    /// Пайплайны ReLU (ленивая инициализация)
    relu_pipelines: OnceLock<ReLUPipelines>,
    /// Пайплайны Sigmoid (ленивая инициализация)
    sigmoid_pipelines: OnceLock<SigmoidPipelines>,
    /// Пайплайны Tanh (ленивая инициализация)
    tanh_pipelines: OnceLock<TanhPipelines>,
    /// Пайплайны LeakyReLU (ленивая инициализация)
    leaky_relu_pipelines: OnceLock<LeakyReLUPipelines>,
    /// Пайплайны Linear (ленивая инициализация)
    linear_pipelines: OnceLock<LinearPipelines>,
    /// Пайплайны SoftSparseGate (ленивая инициализация)
    soft_sparse_gate_pipelines: OnceLock<SoftSparseGatePipelines>,
    /// Пайплайны SoftKeepGate (ленивая инициализация)
    soft_keep_gate_pipelines: OnceLock<SoftKeepGatePipelines>,
    /// Пайплайны DualAnchor (ленивая инициализация)
    dual_anchor_pipelines: OnceLock<DualAnchorPipelines>,
    /// Пайплайны Softmax (ленивая инициализация)
    softmax_pipelines: OnceLock<SoftmaxPipelines>,
    /// Пайплайны Memory (ленивая инициализация)
    memory_pipelines: OnceLock<MemoryPipelines>,
    /// Пайплайны Splitter (ленивая инициализация)
    splitter_pipelines: OnceLock<SplitterPipelines>,
    /// Пайплайны Combiner (ленивая инициализация)
    combiner_pipelines: OnceLock<CombinerPipelines>,
}

impl GpuCompute {
    pub fn new(
        context: Arc<GpuContext>,
        pipeline_cache: Arc<PipelineCache>,
        memory_executor: Arc<Mutex<MemoryExecutor>>,
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
            relu_pipelines: OnceLock::new(),
            sigmoid_pipelines: OnceLock::new(),
            tanh_pipelines: OnceLock::new(),
            leaky_relu_pipelines: OnceLock::new(),
            linear_pipelines: OnceLock::new(),
            soft_sparse_gate_pipelines: OnceLock::new(),
            soft_keep_gate_pipelines: OnceLock::new(),
            dual_anchor_pipelines: OnceLock::new(),
            softmax_pipelines: OnceLock::new(),
            memory_pipelines: OnceLock::new(),
            splitter_pipelines: OnceLock::new(),
            combiner_pipelines: OnceLock::new(),
        }
    }

    /// Получить пайплайны ReLU
    pub fn relu_pipelines(&self) -> &ReLUPipelines {
        self.relu_pipelines
            .get_or_init(|| ReLUPipelines::new(self.context.device.clone()))
    }

    /// Получить пайплайны Sigmoid
    pub fn sigmoid_pipelines(&self) -> &SigmoidPipelines {
        self.sigmoid_pipelines
            .get_or_init(|| SigmoidPipelines::new(self.context.device.clone()))
    }

    /// Получить пайплайны Tanh
    pub fn tanh_pipelines(&self) -> &TanhPipelines {
        self.tanh_pipelines
            .get_or_init(|| TanhPipelines::new(self.context.device.clone()))
    }

    /// Получить пайплайны LeakyReLU
    pub fn leaky_relu_pipelines(&self) -> &LeakyReLUPipelines {
        self.leaky_relu_pipelines
            .get_or_init(|| LeakyReLUPipelines::new(self.context.device.clone()))
    }

    /// Получить пайплайны Linear
    pub fn linear_pipelines(&self) -> &LinearPipelines {
        self.linear_pipelines
            .get_or_init(|| LinearPipelines::new(self.context.device.clone()))
    }

    /// Получить пайплайны SoftSparseGate
    pub fn soft_sparse_gate_pipelines(&self) -> &SoftSparseGatePipelines {
        self.soft_sparse_gate_pipelines
            .get_or_init(|| SoftSparseGatePipelines::new(self.context.device.clone()))
    }

    /// Получить пайплайны SoftKeepGate
    pub fn soft_keep_gate_pipelines(&self) -> &SoftKeepGatePipelines {
        self.soft_keep_gate_pipelines
            .get_or_init(|| SoftKeepGatePipelines::new(self.context.device.clone()))
    }

    /// Получить пайплайны DualAnchor
    pub fn dual_anchor_pipelines(&self) -> &DualAnchorPipelines {
        self.dual_anchor_pipelines
            .get_or_init(|| DualAnchorPipelines::new(self.context.device.clone()))
    }

    /// Получить пайплайны Softmax
    pub fn softmax_pipelines(&self) -> &SoftmaxPipelines {
        self.softmax_pipelines
            .get_or_init(|| SoftmaxPipelines::new(self.context.device.clone()))
    }

    /// Получить пайплайны Memory
    pub fn memory_pipelines(&self) -> &MemoryPipelines {
        self.memory_pipelines
            .get_or_init(|| MemoryPipelines::new(self.context.device.clone()))
    }

    /// Получить пайплайны Splitter
    pub fn splitter_pipelines(&self) -> &SplitterPipelines {
        self.splitter_pipelines
            .get_or_init(|| SplitterPipelines::new(self.context.device.clone()))
    }

    /// Получить пайплайны Combiner
    pub fn combiner_pipelines(&self) -> &CombinerPipelines {
        self.combiner_pipelines
            .get_or_init(|| CombinerPipelines::new(self.context.device.clone()))
    }

    // --- Временные буферы ---

    pub fn acquire_temp_buffer(
        &self,
        elements: usize,
    ) -> (Subbuffer<[f32]>, RawBufferId) {
        let kind = MemoryDeviceKind::DeviceVram(self.gpu_device_id);
        self.memory_executor.lock().unwrap().acquire_temp_buffer(kind, elements)
    }

    pub fn acquire_staging_buffer(
        &self,
        elements: usize,
    ) -> (Subbuffer<[f32]>, RawBufferId) {
        self.memory_executor.lock().unwrap().acquire_temp_buffer(MemoryDeviceKind::HostRam, elements)
    }

    pub fn release_temp_buffer(
        &self,
        buffer: Subbuffer<[f32]>,
        raw_id: RawBufferId,
    ) {
        let kind = MemoryDeviceKind::DeviceVram(self.gpu_device_id);
        self.memory_executor.lock().unwrap().release_temp_buffer(kind, buffer, raw_id);
    }

    pub fn release_staging_buffer(
        &self,
        buffer: Subbuffer<[f32]>,
        raw_id: RawBufferId,
    ) {
        self.memory_executor.lock().unwrap().release_temp_buffer(MemoryDeviceKind::HostRam, buffer, raw_id);
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
        pipeline: Arc<vulkano::pipeline::ComputePipeline>,
        buffers: &[(u32, Subbuffer<[f32]>)],
        push_constants: &[u32; N],
        total_elements: usize,
    ) {
        let dispatch_dim = [((total_elements + 255) / 256) as u32, 1, 1];
        self.run_compute_shader_with_dispatch(pipeline, buffers, push_constants, dispatch_dim);
    }

    pub fn run_compute_shader_with_dispatch<const N: usize>(
        &self,
        pipeline: Arc<vulkano::pipeline::ComputePipeline>,
        buffers: &[(u32, Subbuffer<[f32]>)],
        push_constants: &[u32; N],
        dispatch_dim: [u32; 3],
    ) {
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
        pipeline: Arc<vulkano::pipeline::ComputePipeline>,
        buffers: &[(u32, Subbuffer<[f32]>)],
        push_constants: &[u32; N],
        dispatch_dim: [u32; 3],
    ) {
        self.run_compute_shader_with_dispatch(pipeline, buffers, push_constants, dispatch_dim);
    }

    // ===================================================================
    // МЕТОДЫ ДЛЯ РАБОТЫ С MatrixBufferHandle
    // ===================================================================

    /// Выделяет GPU-буфер через MemoryExecutor (DeviceVram).
    pub fn allocate_gpu_matrix_handle(&self, rows: usize, cols: usize) -> MatrixBufferHandle {
        let mut mem = self.memory_executor.lock().unwrap();
        mem.acquire_matrix_handle(
            rows,
            cols,
            MemoryDeviceKind::DeviceVram(self.gpu_device_id),
            BufferPriority::Medium,
        )
        .expect("Failed to allocate GPU MatrixBufferHandle")
    }

    /// Выделяет CPU-буфер через MemoryExecutor (HostRam).
    pub fn allocate_cpu_matrix_handle(&self, rows: usize, cols: usize) -> MatrixBufferHandle {
        let mut mem = self.memory_executor.lock().unwrap();
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

    pub fn download_gpu_handle_to_vec(&self, handle: &MatrixBufferHandle) -> Vec<f32> {
        assert!(handle.is_gpu(), "Handle must be GPU");
        let elements = handle.rows() * handle.cols();

        let gpu_buf = self.get_gpu_subbuffer_from_handle(handle);
        let (staging_buf, staging_raw) = self.acquire_staging_buffer(elements);
        self.copy_buffer_sync(gpu_buf, staging_buf.clone());

        let data = {
            let guard = staging_buf.read().expect("read staging buffer");
            guard[..elements].to_vec()
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

    /// Копирует регион GPU-буфера с указанием байтовых смещений.
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

        let elem_size = std::mem::size_of::<f32>() as u64;
        let src_start_byte = src_offset as u64 * elem_size;
        let dst_start_byte = dst_offset as u64 * elem_size;
        let byte_len = elements as u64 * elem_size;

        let src_full = self.get_gpu_subbuffer_from_handle(src);
        let dst_full = self.get_gpu_subbuffer_from_handle(dst);

        // Преобразуем в Subbuffer<[u8]> с помощью into_bytes()
        let src_u8 = src_full.into_bytes();
        let dst_u8 = dst_full.into_bytes();

        // Создаём регион копирования с приватным полем _ne через Default
        let region = BufferCopy {
            src_offset: src_start_byte,
            dst_offset: dst_start_byte,
            size: byte_len,
            ..Default::default()
        };

        // Создаём CopyBufferInfo через buffers() и заменяем регионы
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
        let mem = self.memory_executor.lock().unwrap();
        let entry = mem.get_matrix_entry(handle.id())
            .expect("MatrixBufferHandle: entry not found");
        match &entry.storage {
            MatrixStorage::Gpu { buffer, .. } => buffer.clone(),
            _ => panic!("Expected GPU storage for handle"),
        }
    }
}
