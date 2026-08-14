// src/compute_manager/gpu/compute/base.rs

use std::sync::Arc;
use std::sync::Mutex;

use faer::Mat;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    allocator::StandardCommandBufferAllocator,
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
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
use crate::compute_manager::persistent_buffer::DeviceBuffer;
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

    fn resolve_persistent_to_subbuffer(
        &self,
        _buffer: &DeviceBuffer,
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

    // --- Одномерный диспатч ---

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

    // --- Явный диспатч ---

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

    // --- Двумерный диспатч ---

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
    // НОВЫЕ МЕТОДЫ ДЛЯ РАБОТЫ С MatrixBufferHandle
    // ===================================================================

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

    pub fn download_gpu_handle_to_mat(&self, handle: &MatrixBufferHandle) -> Mat<f32> {
        let rows = handle.rows();
        let cols = handle.cols();
        let vec = self.download_gpu_handle_to_vec(handle);
        Mat::from_fn(rows, cols, |r, c| vec[c * rows + r])
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

        let data_vec = self.download_gpu_handle_to_vec(src);

        let mut dst_guard = dst.write();
        let dst_slice = dst_guard.as_slice_mut().expect("Destination is not CPU");
        dst_slice.copy_from_slice(&data_vec);
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

        let elem_size = std::mem::size_of::<f32>() as u64;
        let src_start_byte = src_offset as u64 * elem_size;
        let dst_start_byte = dst_offset as u64 * elem_size;
        let byte_len = elements as u64 * elem_size;

        let src_full = self.get_gpu_subbuffer_from_handle(src);
        let dst_full = self.get_gpu_subbuffer_from_handle(dst);

        let src_slice = src_full.clone().slice(src_start_byte..(src_start_byte + byte_len));
        let dst_slice = dst_full.clone().slice(dst_start_byte..(dst_start_byte + byte_len));

        self.copy_buffer_sync(src_slice, dst_slice);
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