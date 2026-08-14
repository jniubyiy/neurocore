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
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::compute_manager::memory_executor::matrix_entry::MatrixStorage;

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

    // --- Загрузка / выгрузка данных (старые методы для MatrixBuffer и Mat) ---

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

    pub fn copy_persistent_to_persistent(
        &self,
        src: &DeviceBuffer,
        dst: &DeviceBuffer,
    ) {
        let src_buf = self.resolve_persistent_to_subbuffer(src);
        let dst_buf = self.resolve_persistent_to_subbuffer(dst);
        self.copy_buffer_sync(src_buf, dst_buf);
    }

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
    /// В текущей архитектуре DeviceBuffer не хранит сам Subbuffer, поэтому метод
    /// остаётся заглушкой для будущей доработки. Для операций с persistent буферами
    /// используйте специализированные методы, если они реализованы.
    fn resolve_persistent_to_subbuffer(
        &self,
        buffer: &DeviceBuffer,
    ) -> Subbuffer<[f32]> {
        panic!("resolve_persistent_to_subbuffer needs refactoring: DeviceBuffer must store Subbuffer");
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
    // МЕТОДЫ ДЛЯ РАБОТЫ С MatrixBuffer (старая система)
    // ===================================================================

    pub fn allocate_gpu_matrix(&self, rows: usize, cols: usize) -> MatrixBuffer {
        MatrixBuffer::new_gpu(&self.memory_executor, self.gpu_device_id, rows, cols)
            .expect("Failed to allocate GPU MatrixBuffer")
    }

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

    pub fn download_gpu_matrix_to_vec(&self, buf: &MatrixBuffer) -> Vec<f32> {
        assert!(buf.is_gpu(), "Buffer must be GPU");
        let elements = buf.size();

        let src_gpu = buf.as_gpu_buffer().expect("GPU buffer");
        let (staging_buf, staging_raw) = self.acquire_staging_buffer(elements);
        self.copy_buffer_sync(src_gpu.clone(), staging_buf.clone());

        let data = {
            let guard = staging_buf.read().expect("read staging buffer");
            guard[..elements].to_vec()
        };
        self.release_staging_buffer(staging_buf, staging_raw);
        data
    }

    pub fn upload_vec_to_gpu_buffer(
        &self,
        data: &[f32],
        rows: usize,
        cols: usize,
    ) -> MatrixBuffer {
        assert_eq!(data.len(), rows * cols, "Data length must match matrix size");

        let mut gpu_buf = self.allocate_gpu_matrix(rows, cols);

        let (staging_buf, staging_raw) = self.acquire_staging_buffer(data.len());
        {
            let mut write_guard = staging_buf.write().expect("write staging buffer");
            write_guard[..data.len()].copy_from_slice(data);
        }
        let dst_gpu = gpu_buf.as_gpu_buffer().expect("GPU buffer");
        self.copy_buffer_sync(staging_buf.clone(), dst_gpu.clone());
        self.release_staging_buffer(staging_buf, staging_raw);

        gpu_buf
    }

    pub fn fill_gpu_buffer(&self, buf: &mut MatrixBuffer, value: f32) {
        assert!(buf.is_gpu(), "Buffer must be GPU");
        let elements = buf.size();
        let data = vec![value; elements];

        let (staging_buf, staging_raw) = self.acquire_staging_buffer(elements);
        {
            let mut write_guard = staging_buf.write().expect("write staging buffer");
            write_guard[..elements].copy_from_slice(&data);
        }
        let dst_gpu = buf.as_gpu_buffer().expect("GPU buffer");
        self.copy_buffer_sync(staging_buf.clone(), dst_gpu.clone());
        self.release_staging_buffer(staging_buf, staging_raw);
    }

    pub fn copy_gpu_submatrix(
        &self,
        src: &MatrixBuffer,
        dst: &mut MatrixBuffer,
        src_row_start: usize,
        src_col_start: usize,
        dst_row_start: usize,
        dst_col_start: usize,
        rows: usize,
        cols: usize,
    ) {
        assert!(src.is_gpu() && dst.is_gpu(), "Both buffers must be GPU");
        let src_rows = src.rows();
        let dst_rows = dst.rows();

        let src_vec = self.download_gpu_matrix_to_vec(src);
        let mut dst_vec = self.download_gpu_matrix_to_vec(dst);

        for c in 0..cols {
            for r in 0..rows {
                let src_idx = (src_col_start + c) * src_rows + (src_row_start + r);
                let dst_idx = (dst_col_start + c) * dst_rows + (dst_row_start + r);
                dst_vec[dst_idx] = src_vec[src_idx];
            }
        }

        let new_dst = self.upload_vec_to_gpu_buffer(&dst_vec, dst_rows, dst.cols());
        *dst = new_dst;
    }

    pub fn concat_gpu_buffers(&self, a: &MatrixBuffer, b: &MatrixBuffer) -> MatrixBuffer {
        assert!(a.is_gpu() && b.is_gpu(), "Both buffers must be GPU");
        assert_eq!(a.rows(), b.rows(), "Row counts must match for concatenation");
        let rows = a.rows();

        let a_vec = self.download_gpu_matrix_to_vec(a);
        let b_vec = self.download_gpu_matrix_to_vec(b);
        let mut combined = a_vec;
        combined.extend_from_slice(&b_vec);

        self.upload_vec_to_gpu_buffer(&combined, rows, a.cols() + b.cols())
    }

    pub fn broadcast_gpu_buffer(&self, vec_buf: &MatrixBuffer, total_cols: usize) -> MatrixBuffer {
        assert!(vec_buf.is_gpu(), "Input must be GPU buffer");
        assert_eq!(vec_buf.cols(), 1, "Input must be a single column vector");
        let rows = vec_buf.rows();

        let vec = self.download_gpu_matrix_to_vec(vec_buf);
        let mut result = Vec::with_capacity(rows * total_cols);
        for _ in 0..total_cols {
            result.extend_from_slice(&vec);
        }

        self.upload_vec_to_gpu_buffer(&result, rows, total_cols)
    }

    pub fn transpose_gpu_matrix(&self, mat: &MatrixBuffer) -> MatrixBuffer {
        assert!(mat.is_gpu(), "Input must be GPU buffer");
        let rows = mat.rows();
        let cols = mat.cols();
        let data = self.download_gpu_matrix_to_vec(mat);

        let mut transposed = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                transposed[r * cols + c] = data[c * rows + r];
            }
        }

        self.upload_vec_to_gpu_buffer(&transposed, cols, rows)
    }

    // ===================================================================
    // НОВЫЕ МЕТОДЫ ДЛЯ РАБОТЫ С MatrixBufferHandle
    // ===================================================================

    /// Создаёт новый GPU MatrixBufferHandle и регистрирует его в MemoryExecutor.
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

    /// Загружает вектор f32 в новый GPU MatrixBufferHandle указанной формы.
    /// Данные ожидаются в column-major порядке (как в MatrixBuffer).
    pub fn upload_vec_to_gpu_handle(
        &self,
        data: &[f32],
        rows: usize,
        cols: usize,
    ) -> MatrixBufferHandle {
        assert_eq!(data.len(), rows * cols, "Data length must match matrix size");
        let mut gpu_handle = self.allocate_gpu_matrix_handle(rows, cols);

        // Копируем данные из CPU в GPU через staging
        self.copy_slice_to_gpu_handle(&gpu_handle, data);

        gpu_handle
    }

    /// Копирует данные из слайса в существующий GPU MatrixBufferHandle.
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

    /// Скачивает содержимое GPU MatrixBufferHandle в обычный вектор f32.
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

    /// Скачивает GPU MatrixBufferHandle в `Mat<f32>`.
    pub fn download_gpu_handle_to_mat(&self, handle: &MatrixBufferHandle) -> Mat<f32> {
        let rows = handle.rows();
        let cols = handle.cols();
        let vec = self.download_gpu_handle_to_vec(handle);
        Mat::from_fn(rows, cols, |r, c| vec[c * rows + r])
    }

    /// Заполняет GPU MatrixBufferHandle заданным значением.
    pub fn fill_gpu_handle(&self, handle: &MatrixBufferHandle, value: f32) {
        let elements = handle.rows() * handle.cols();
        let data = vec![value; elements];
        self.copy_slice_to_gpu_handle(handle, &data);
    }

    /// Копирует данные из CPU MatrixBufferHandle в GPU MatrixBufferHandle.
    /// Источник должен быть CPU, назначение GPU.
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

    /// Копирует данные из GPU MatrixBufferHandle в CPU MatrixBufferHandle.
    pub fn copy_gpu_to_cpu_handle(
        &self,
        src: &MatrixBufferHandle,
        dst: &MatrixBufferHandle,
    ) {
        assert!(src.is_gpu(), "Source must be GPU");
        assert!(!dst.is_gpu(), "Destination must be CPU");
        let elements = src.rows() * src.cols();
        assert_eq!(elements, dst.rows() * dst.cols(), "Buffer sizes must match");

        let data_vec = self.download_gpu_handle_to_vec(src);

        let mut dst_guard = dst.write();
        let dst_slice = dst_guard.as_slice_mut().expect("Destination is not CPU");
        dst_slice.copy_from_slice(&data_vec);
    }

    /// Вспомогательный метод: извлекает Subbuffer из GPU-записи по дескриптору.
    /// Блокировка MemoryExecutor снимается после получения клона Subbuffer.
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