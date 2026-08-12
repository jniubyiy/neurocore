// src/compute_manager/gpu/compute/base.rs

use std::sync::Arc;
use std::sync::Mutex;

use faer::Mat;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    allocator::StandardCommandBufferAllocator,
    AutoCommandBufferBuilder, CommandBufferUsage,
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
    TensorBufferId,
    BufferPriority,
};
use crate::compute_manager::persistent_buffer::{DeviceBuffer, DeviceBufferId};
use crate::compute_manager::matrix_buffer::MatrixBuffer;
use super::super::init::GpuContext;
use super::super::pipeline::PipelineCache;

pub struct GpuCompute {
    pub context: Arc<GpuContext>,
    pub pipeline_cache: Arc<PipelineCache>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub memory_executor: Arc<Mutex<MemoryExecutor>>,
    pub gpu_device_id: DeviceId,
    pub memory_state: Option<Subbuffer<[f32]>>,
    pub memory_state_id: Option<RawBufferId>,
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
            memory_state: None,
            memory_state_id: None,
        }
    }

    // --- Долгоживущие буферы (для совместимости с моделью) ---

    pub fn create_buffer(
        &self,
        elements: usize,
        _usage: BufferUsage,
    ) -> (Subbuffer<[f32]>, TensorBufferId) {
        let mut mem = self.memory_executor.lock().unwrap();
        let kind = MemoryDeviceKind::DeviceVram(self.gpu_device_id);
        let id = mem.allocate(kind, elements, BufferPriority::High)
            .expect("Failed to allocate GPU buffer");
        let resolved = mem.resolve_buffer(id, kind)
            .expect("Failed to resolve buffer");
        (resolved.as_device_buffer().clone(), id)
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

    // --- Загрузка / выгрузка данных ---

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

    pub fn read_temp_buffer_to_mat(
        &self,
        buffer: Subbuffer<[f32]>,
        raw_id: RawBufferId,
        rows: usize,
        cols: usize,
    ) -> Mat<f32> {
        let total = rows * cols;
        let (staging_buf, staging_raw) = self.acquire_staging_buffer(total);
        self.copy_buffer_sync(buffer.clone(), staging_buf.clone());

        let data_vec = {
            let guard = staging_buf.read().expect("read staging buffer");
            let slice = &guard[..total];
            slice.to_vec()
        };

        self.release_staging_buffer(staging_buf, staging_raw);
        self.release_temp_buffer(buffer, raw_id);

        Mat::from_fn(rows, cols, |r, c| data_vec[r * cols + c])
    }

    // --- Работа с постоянными (persistent) буферами ---

    /// Копирует данные из одного persistent буфера в другой (оба должны быть на GPU).
    /// Использует временный staging-буфер в VRAM (или напрямую копирует, если поддерживается).
    pub fn copy_persistent_to_persistent(
        &self,
        src: &DeviceBuffer,
        dst: &DeviceBuffer,
    ) {
        let src_buf = self.resolve_persistent_to_subbuffer(src);
        let dst_buf = self.resolve_persistent_to_subbuffer(dst);
        self.copy_buffer_sync(src_buf, dst_buf);
    }

    /// Загружает данные CPU (срез f32) в постоянный GPU-буфер.
    pub fn fill_persistent_buffer(
        &self,
        buffer: &DeviceBuffer,
        data: &[f32],
    ) {
        let elements = buffer.size_elements;
        assert_eq!(data.len(), elements, "Data size must match buffer size");

        let (staging_buf, staging_raw) = self.acquire_staging_buffer(elements);
        {
            let mut write_guard = staging_buf.write().expect("write staging buffer");
            write_guard[..elements].copy_from_slice(data);
        }
        let dst_buf = self.resolve_persistent_to_subbuffer(buffer);
        self.copy_buffer_sync(staging_buf.clone(), dst_buf);
        self.release_staging_buffer(staging_buf, staging_raw);
    }

    /// Выгружает данные из постоянного GPU-буфера в вектор f32 на CPU.
    pub fn read_persistent_buffer(
        &self,
        buffer: &DeviceBuffer,
    ) -> Vec<f32> {
        let elements = buffer.size_elements;
        let (staging_buf, staging_raw) = self.acquire_staging_buffer(elements);
        let src_buf = self.resolve_persistent_to_subbuffer(buffer);
        self.copy_buffer_sync(src_buf, staging_buf.clone());

        let data_vec = {
            let guard = staging_buf.read().expect("read staging buffer");
            let slice = &guard[..elements];
            slice.to_vec()
        };
        self.release_staging_buffer(staging_buf, staging_raw);
        data_vec
    }

    /// Вспомогательный метод: по идентификатору persistent буфера возвращает Subbuffer.
    fn resolve_persistent_to_subbuffer(
        &self,
        buffer: &DeviceBuffer,
    ) -> Subbuffer<[f32]> {
        match &buffer.id {
            DeviceBufferId::Gpu(raw_id) => {
                // Получаем доступ к raw буферу через MemoryExecutor.
                // Нам нужен Subbuffer, но raw_registry только хранит метаданные.
                // Поэтому мы создаём временный "псевдо"-Subbuffer, используя тот же raw_id?
                // На самом деле Subbuffer – это вулкановский буфер с known size.
                // Мы не можем извлечь его напрямую из raw_registry, так как там нет Subbuffer.
                // Вместо этого, мы должны хранить Subbuffer где-то вместе с persistent буфером.
                // Это ограничение текущей архитектуры. Нужно дополнить DeviceBuffer хранением Subbuffer.
                // Пока оставим заглушку: будем требовать, чтобы DeviceBuffer хранил сам Subbuffer,
                // либо передавать Subbuffer вместе с буфером.
                // Для простоты сейчас добавим метод `buffer()` в DeviceBuffer.
                // Но т.к. DeviceBuffer определён в persistent_buffer.rs, мы должны его изменить.
                // Временно вызовем панику, а реальная реализация потребует рефакторинга DeviceBuffer.
                panic!("resolve_persistent_to_subbuffer needs refactoring: DeviceBuffer must store Subbuffer");
            },
            _ => panic!("Persistent buffer is not on GPU"),
        }
    }

    // --- Копирование между двумя Subbuffer'ами (синхронно) ---
    pub fn copy_buffer_sync(&self, src: Subbuffer<[f32]>, dst: Subbuffer<[f32]>) {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .copy_buffer(vulkano::command_buffer::CopyBufferInfo::buffers(src, dst))
            .unwrap();
        let cb = builder.build().unwrap();
        let future = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
    }

    // --- Одномерный диспатч (для активаций, loss-кубиков) ---

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

    // --- Явный диспатч (для softmax, cross-entropy, matmul) ---

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

    // --- Двумерный диспатч для matmul (оставлен для обратной совместимости) ---

    pub fn run_compute_shader_2d<const N: usize>(
        &self,
        pipeline: Arc<vulkano::pipeline::ComputePipeline>,
        buffers: &[(u32, Subbuffer<[f32]>)],
        push_constants: &[u32; N],
        dispatch_dim: [u32; 3],
    ) {
        self.run_compute_shader_with_dispatch(pipeline, buffers, push_constants, dispatch_dim);
    }

    pub fn mat_to_flat(mat: &Mat<f32>) -> Vec<f32> {
        let rows = mat.nrows();
        let cols = mat.ncols();
        let mut flat = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                flat.push(mat[(r, c)]);
            }
        }
        flat
    }

    // ===================================================================
    // НОВЫЕ МЕТОДЫ ДЛЯ РАБОТЫ С MatrixBuffer (GPU)
    // ===================================================================

    /// Создаёт GPU MatrixBuffer нужного размера.
    pub fn allocate_gpu_matrix(&self, rows: usize, cols: usize) -> MatrixBuffer {
        MatrixBuffer::new_gpu(&self.memory_executor, self.gpu_device_id, rows, cols)
            .expect("Failed to allocate GPU MatrixBuffer")
    }

    /// Копирует данные из CPU MatrixBuffer в GPU MatrixBuffer.
    /// Предполагается, что `src` – CPU, `dst` – GPU.
    pub fn copy_cpu_to_gpu(&self, src: &MatrixBuffer, dst: &mut MatrixBuffer) {
        assert!(!src.is_gpu(), "Source must be CPU");
        assert!(dst.is_gpu(), "Destination must be GPU");
        let elements = src.size();
        assert_eq!(elements, dst.size(), "Buffer sizes must match");

        let src_slice = src.as_slice();
        let (staging_buf, staging_raw) = self.acquire_staging_buffer(elements);
        {
            let mut write_guard = staging_buf.write().expect("write staging buffer");
            write_guard[..elements].copy_from_slice(src_slice);
        }
        let dst_gpu = dst.as_gpu_buffer().expect("Destination is GPU");
        self.copy_buffer_sync(staging_buf.clone(), dst_gpu.clone());
        self.release_staging_buffer(staging_buf, staging_raw);
    }

    /// Копирует данные из GPU MatrixBuffer в CPU MatrixBuffer.
    /// Предполагается, что `src` – GPU, `dst` – CPU.
    pub fn copy_gpu_to_cpu(&self, src: &MatrixBuffer, dst: &mut MatrixBuffer) {
        assert!(src.is_gpu(), "Source must be GPU");
        assert!(!dst.is_gpu(), "Destination must be CPU");
        let elements = src.size();
        assert_eq!(elements, dst.size(), "Buffer sizes must match");

        let src_gpu = src.as_gpu_buffer().expect("Source is GPU");
        let (staging_buf, staging_raw) = self.acquire_staging_buffer(elements);
        self.copy_buffer_sync(src_gpu.clone(), staging_buf.clone());

        let data_vec = {
            let guard = staging_buf.read().expect("read staging buffer");
            guard[..elements].to_vec()
        };
        self.release_staging_buffer(staging_buf, staging_raw);

        dst.copy_from_slice(&data_vec);
    }

    /// Удобный метод: загружает Mat в GPU MatrixBuffer.
    pub fn upload_mat_to_gpu_matrix(&self, mat: &Mat<f32>) -> MatrixBuffer {
        let rows = mat.nrows();
        let cols = mat.ncols();
        let mut gpu_buf = self.allocate_gpu_matrix(rows, cols);

        let flat = Self::mat_to_flat(mat);
        let (staging_buf, staging_raw) = self.acquire_staging_buffer(flat.len());
        {
            let mut write_guard = staging_buf.write().expect("write staging buffer");
            write_guard[..flat.len()].copy_from_slice(&flat);
        }
        let dst_gpu = gpu_buf.as_gpu_buffer().expect("GPU buffer");
        self.copy_buffer_sync(staging_buf.clone(), dst_gpu.clone());
        self.release_staging_buffer(staging_buf, staging_raw);

        gpu_buf
    }

    /// Удобный метод: выгружает GPU MatrixBuffer в Mat.
    pub fn download_gpu_matrix_to_mat(&self, buf: &MatrixBuffer) -> Mat<f32> {
        assert!(buf.is_gpu(), "Buffer must be GPU");
        let rows = buf.rows();
        let cols = buf.cols();
        let elements = buf.size();

        let src_gpu = buf.as_gpu_buffer().expect("GPU buffer");
        let (staging_buf, staging_raw) = self.acquire_staging_buffer(elements);
        self.copy_buffer_sync(src_gpu.clone(), staging_buf.clone());

        let data_vec = {
            let guard = staging_buf.read().expect("read staging buffer");
            guard[..elements].to_vec()
        };
        self.release_staging_buffer(staging_buf, staging_raw);

        Mat::from_fn(rows, cols, |r, c| data_vec[r * cols + c])
    }
}