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
    TensorBufferId,
    BufferPriority,
};
use crate::compute_manager::logger;

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
    pub memory_state_id: Option<TensorBufferId>,
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

    /// Выделить буфер в VRAM через MemoryExecutor. Возвращает Subbuffer и ID для освобождения.
    pub fn create_buffer(
        &self,
        elements: usize,
        _usage: BufferUsage,
    ) -> (Subbuffer<[f32]>, TensorBufferId) {
        let mut mem = self.memory_executor.lock().unwrap();
        let kind = MemoryDeviceKind::DeviceVram(self.gpu_device_id);
        let id = mem.allocate(kind, elements, BufferPriority::High)
            .expect("Failed to allocate GPU buffer");
        logger::log(format!("[GPU] create_buffer: id={}, elements={}", id.0, elements));
        let resolved = mem.resolve_buffer(id, kind)
            .expect("Failed to resolve buffer");
        let buf = resolved.as_device_buffer().clone();
        drop(resolved);
        (buf, id)
    }

    pub fn release_buffer(&self, id: TensorBufferId) {
        logger::log(format!("[GPU] release_buffer: id={}", id.0));
        self.memory_executor.lock().unwrap().release_buffer(id);
    }

    /// Создать буфер в DeviceVram и заполнить данными из CPU.
    pub fn create_storage_buffer_from_slice(
        &self,
        data: &[f32],
    ) -> (Subbuffer<[f32]>, TensorBufferId) {
        let elements = data.len();
        logger::log(format!("[GPU] create_storage_buffer_from_slice: {} elems", elements));
        let mut mem = self.memory_executor.lock().unwrap();

        // 1. Выделяем HostRam буфер и записываем данные
        let host_id = mem.allocate(MemoryDeviceKind::HostRam, elements, BufferPriority::High)
            .expect("Failed to allocate host buffer");
        {
            let mut resolved = mem.resolve_buffer(host_id, MemoryDeviceKind::HostRam)
                .expect("Failed to resolve host buffer");
            resolved.as_host_slice_mut().copy_from_slice(data);
        }

        // 2. Перемещаем буфер в DeviceVram
        mem.move_buffer(host_id, MemoryDeviceKind::DeviceVram(self.gpu_device_id))
            .expect("Failed to move buffer to device");

        // 3. Получаем Subbuffer
        let resolved = mem.resolve_buffer(host_id, MemoryDeviceKind::DeviceVram(self.gpu_device_id))
            .expect("Failed to resolve device buffer");
        let buf = resolved.as_device_buffer().clone();
        drop(resolved);
        (buf, host_id)
    }

    /// Копирует данные из одного буфера в другой (синхронно).
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

    /// Запуск шейдера с 1 входом и 1 выходом.
    pub fn run_elementwise_1in_1out<const N: usize>(
        &self,
        pipeline: Arc<vulkano::pipeline::ComputePipeline>,
        input: Subbuffer<[f32]>,
        output_elements: usize,
        push_data: [u32; N],
    ) -> (Subbuffer<[f32]>, TensorBufferId) {
        let (out_buf, out_id) = self.create_buffer(
            output_elements,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        );

        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input.clone()),
                WriteDescriptorSet::buffer(1, out_buf.clone()),
            ],
            [],
        )
        .expect("descriptor set");

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("command buffer builder");

        let dispatch_dim = [((output_elements + 255) / 256) as u32, 1, 1];

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
                .push_constants(pipeline.layout().clone(), 0, push_data)
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

        (out_buf, out_id)
    }

    /// Запуск шейдера с 2 входами и 1 выходом.
    pub fn run_elementwise_2in_1out<const N: usize>(
        &self,
        pipeline: Arc<vulkano::pipeline::ComputePipeline>,
        input_a: Subbuffer<[f32]>,
        input_b: Subbuffer<[f32]>,
        output_elements: usize,
        push_data: [u32; N],
    ) -> (Subbuffer<[f32]>, TensorBufferId) {
        let (out_buf, out_id) = self.create_buffer(
            output_elements,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        );

        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input_a.clone()),
                WriteDescriptorSet::buffer(1, input_b.clone()),
                WriteDescriptorSet::buffer(2, out_buf.clone()),
            ],
            [],
        )
        .expect("descriptor set");

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("command buffer builder");

        let dispatch_dim = [((output_elements + 255) / 256) as u32, 1, 1];

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
                .push_constants(pipeline.layout().clone(), 0, push_data)
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

        (out_buf, out_id)
    }

    /// Читает GPU-буфер в матрицу и освобождает буфер.
    pub fn read_buffer_to_mat(
        &self,
        _buffer: Subbuffer<[f32]>,   // больше не используется
        buffer_id: TensorBufferId,
        rows: usize,
        cols: usize,
    ) -> Mat<f32> {
        let total = rows * cols;
        logger::log(format!(
            "[GPU] read_buffer_to_mat: buffer_id={}, rows={}, cols={} ({} elems)",
            buffer_id.0, rows, cols, total
        ));

        // 1. Перемещаем буфер в HostRam через MemoryExecutor
        {
            let mut mem = self.memory_executor.lock().unwrap();
            mem.move_buffer(buffer_id, MemoryDeviceKind::HostRam)
                .expect("Failed to move buffer to host for reading");
        }

        // 2. Читаем данные напрямую из HostRam-представления
        let data_vec = {
            let mut mem = self.memory_executor.lock().unwrap();
            let resolved = mem.resolve_buffer(buffer_id, MemoryDeviceKind::HostRam)
                .expect("Failed to resolve host buffer");
            let slice = resolved.as_host_slice();
            logger::log(format!(
                "[GPU] read_buffer_to_mat: read {} bytes, first values: {:?}",
                slice.len() * 4,
                &slice[..total.min(4)]
            ));
            slice.to_vec()
        };

        // 3. Освобождаем буфер
        self.release_buffer(buffer_id);

        Mat::from_fn(rows, cols, |r, c| data_vec[r * cols + c])
    }
}