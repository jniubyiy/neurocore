// src/compute_manager/gpu/compute/base/handle_ops.rs
//
//! Операции с управляемыми `MatrixBufferHandle`: аллокация, загрузка/выгрузка
//! данных, копирование регионов между GPU-буферами.

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, BufferCopy, CommandBufferUsage, CopyBufferInfo,
};
use vulkano::sync::{self, GpuFuture};

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::compute_manager::memory_executor::{
    matrix_entry::MatrixStorage,
    policy::BufferPriority,
    types::MemoryDeviceKind,
};

use super::GpuCompute;

impl GpuCompute {
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