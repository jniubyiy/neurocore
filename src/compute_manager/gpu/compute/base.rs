// src/compute_manager/gpu/compute/base.rs

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use faer::Mat;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    allocator::StandardCommandBufferAllocator,
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
};
use vulkano::descriptor_set::{
    allocator::StandardDescriptorSetAllocator,
    DescriptorSet, WriteDescriptorSet,
};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};

use super::super::init::GpuContext;
use super::super::pipeline::PipelineCache;

/// Ключ для кэша буферов: (количество элементов f32, тип использования).
#[derive(Hash, Eq, PartialEq, Clone, Copy)]
enum BufferCacheKey {
    /// Буфер для переноса данных (TRANSFER_DST).
    Staging { elements: usize },
    /// Буфер для вычислений (STORAGE_BUFFER | TRANSFER_SRC).
    Compute { elements: usize },
}

impl BufferCacheKey {
    fn elements(&self) -> usize {
        match self {
            BufferCacheKey::Staging { elements } => *elements,
            BufferCacheKey::Compute { elements } => *elements,
        }
    }
}

pub struct GpuCompute {
    pub context: Arc<GpuContext>,
    pub pipeline_cache: Arc<PipelineCache>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub param_buffer: Option<Subbuffer<[f32]>>,
    /// Кэш временных буферов (staging и compute) для повторного использования.
    buffer_cache: RefCell<HashMap<BufferCacheKey, Vec<Subbuffer<[f32]>>>>,
    pub memory_state: Option<Subbuffer<[f32]>>,
}

impl GpuCompute {
    pub fn new(context: Arc<GpuContext>, pipeline_cache: Arc<PipelineCache>) -> Self {
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
            param_buffer: None,
            buffer_cache: RefCell::new(HashMap::new()),
            memory_state: None,
        }
    }

    pub fn upload_params(&mut self, params: &[f32]) {
        self.param_buffer = Some(
            Buffer::from_iter(
                self.context.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                params.iter().copied(),
            )
            .expect("Failed to upload parameters to GPU"),
        );
    }

    /// Создаёт или извлекает из кэша буфер заданного размера и назначения.
    pub fn create_buffer(&self, elements: usize, usage: BufferUsage) -> Subbuffer<[f32]> {
        let key = if usage == BufferUsage::TRANSFER_DST {
            BufferCacheKey::Staging { elements }
        } else {
            BufferCacheKey::Compute { elements }
        };

        // Пытаемся взять из кэша
        if let Some(buf) = self.buffer_cache.borrow_mut().get_mut(&key).and_then(|v| v.pop()) {
            return buf;
        }

        // Иначе создаём новый
        let size = (elements * std::mem::size_of::<f32>()) as u64;
        Buffer::new_unsized(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            size,
        )
        .expect("Failed to create buffer")
    }

    /// Возвращает буфер в кэш для последующего использования.
    fn release_buffer(&self, key: BufferCacheKey, buffer: Subbuffer<[f32]>) {
        debug_assert_eq!(buffer.len() as usize, key.elements());
        self.buffer_cache
            .borrow_mut()
            .entry(key)
            .or_insert_with(Vec::new)
            .push(buffer);
    }

    /// Копирует данные из src в dst (синхронно).
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

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    pub fn create_storage_buffer_from_slice(
        allocator: &Arc<vulkano::memory::allocator::StandardMemoryAllocator>,
        data: &[f32],
    ) -> Subbuffer<[f32]> {
        Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.iter().copied(),
        )
        .expect("Failed to create storage buffer")
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

    /// Универсальный запуск поэлементного шейдера с одним входным буфером и одним выходным.
    pub fn run_elementwise_1in_1out<const N: usize>(
        &self,
        pipeline: Arc<vulkano::pipeline::ComputePipeline>,
        input: Subbuffer<[f32]>,
        output_elements: usize,
        push_data: [u32; N],
    ) -> Subbuffer<[f32]> {
        // Выходной буфер получаем из кэша или создаём
        let out_buf = self.create_buffer(
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

        out_buf
    }

    /// Универсальный запуск поэлементного шейдера с двумя входными буферами и одним выходным.
    pub fn run_elementwise_2in_1out<const N: usize>(
        &self,
        pipeline: Arc<vulkano::pipeline::ComputePipeline>,
        input_a: Subbuffer<[f32]>,
        input_b: Subbuffer<[f32]>,
        output_elements: usize,
        push_data: [u32; N],
    ) -> Subbuffer<[f32]> {
        let out_buf = self.create_buffer(
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

        out_buf
    }

    /// Читает данные из GPU-буфера в матрицу.
    /// Переданный `buffer` после чтения возвращается в кэш, так как он больше не нужен.
    pub fn read_buffer_to_mat(&self, buffer: Subbuffer<[f32]>, rows: usize, cols: usize) -> Mat<f32> {
        let total = rows * cols;

        // Staging-буфер берём из кэша
        let staging = self.create_buffer(total, BufferUsage::TRANSFER_DST);
        self.copy_buffer_sync(buffer.clone(), staging.clone());

        // Читаем данные, копируя их в Vec<f32>, чтобы освободить BufferReadGuard до перемещения staging.
        let data_vec = {
            let guard = staging.read().unwrap();
            let len = guard.len();
            // собираем в собственный вектор
            let mut v = Vec::with_capacity(len);
            v.extend_from_slice(&guard);
            v
        }; // здесь guard дропается

        let mat = Mat::from_fn(rows, cols, |r, c| data_vec[r * cols + c]);

        // Возвращаем оба буфера в кэш
        self.release_buffer(BufferCacheKey::Staging { elements: total }, staging);
        self.release_buffer(
            BufferCacheKey::Compute {
                elements: buffer.len() as usize,
            },
            buffer,
        );

        mat
    }
}