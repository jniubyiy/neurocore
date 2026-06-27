// src/compute_manager/gpu/compute.rs

use std::sync::Arc;

use faer::Mat;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
};
use vulkano::descriptor_set::{
    allocator::StandardDescriptorSetAllocator,
    DescriptorSet, WriteDescriptorSet,
};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};

use super::init::GpuContext;
use super::pipeline::PipelineCache;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;

pub struct GpuCompute {
    context: Arc<GpuContext>,
    pipeline_cache: Arc<PipelineCache>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    param_buffer: Option<Subbuffer<[f32]>>,
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

    // ---------- Linear ----------
    pub fn run_linear_forward(
        &self,
        input: &Mat<f32>,
        weight: &Mat<f32>,
        bias: &[f32],
    ) -> Mat<f32> {
        let batch = input.nrows();
        let in_features = input.ncols();
        let out_features = weight.nrows();
        assert_eq!(weight.ncols(), in_features);
        assert_eq!(bias.len(), out_features);

        let input_buf = Self::create_storage_buffer_from_slice(
            &self.context.memory_allocator,
            &(0..batch)
                .flat_map(|r| (0..in_features).map(move |c| input[(r, c)]))
                .collect::<Vec<f32>>(),
        );

        let weight_transposed: Vec<f32> = (0..in_features)
            .flat_map(|i| (0..out_features).map(move |j| weight[(j, i)]))
            .collect();
        let weight_buf = Self::create_storage_buffer_from_slice(
            &self.context.memory_allocator,
            &weight_transposed,
        );

        let output_size = (batch * out_features) as u64 * std::mem::size_of::<f32>() as u64;
        let output_buf: Subbuffer<[f32]> = Buffer::new_unsized(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            output_size,
        )
        .expect("output buffer");

        let pipeline = self.pipeline_cache.mat_mul_pipeline();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input_buf.clone()),
                WriteDescriptorSet::buffer(1, weight_buf.clone()),
                WriteDescriptorSet::buffer(2, output_buf.clone()),
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

        let dispatch_dim = [
            ((batch + 15) / 16) as u32,
            ((out_features + 15) / 16) as u32,
            1u32,
        ];
        let push_constants: [u32; 3] = [batch as u32, out_features as u32, in_features as u32];

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
                .push_constants(pipeline.layout().clone(), 0, push_constants)
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

        let staging_size = (batch * out_features) as u64 * std::mem::size_of::<f32>() as u64;
        let staging_buf: Subbuffer<[f32]> = Buffer::new_unsized(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            staging_size,
        )
        .expect("staging buffer");

        let mut copy_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        copy_builder
            .copy_buffer(CopyBufferInfo::buffers(output_buf, staging_buf.clone()))
            .unwrap();
        let copy_cb = copy_builder.build().unwrap();
        let future2 = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), copy_cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future2.wait(None).unwrap();

        let data = staging_buf.read().expect("read staging buffer");
        let mut result = Mat::zeros(batch, out_features);
        for r in 0..batch {
            for c in 0..out_features {
                result[(r, c)] = data[r * out_features + c] + bias[c];
            }
        }
        result
    }

    // ---------- Общая функция активации ----------
    fn run_activation_forward(
        &self,
        input: &Mat<f32>,
        op_type: u32,
        alpha: f32,
    ) -> Mat<f32> {
        let batch = input.nrows();
        let features = input.ncols();
        let total_elements = batch * features;

        let input_buf = Self::create_storage_buffer_from_slice(
            &self.context.memory_allocator,
            &(0..batch)
                .flat_map(|r| (0..features).map(move |c| input[(r, c)]))
                .collect::<Vec<f32>>(),
        );

        let output_size = total_elements as u64 * std::mem::size_of::<f32>() as u64;
        let output_buf: Subbuffer<[f32]> = Buffer::new_unsized(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            output_size,
        )
        .expect("output buffer");

        let pipeline = self.pipeline_cache.activation_pipeline();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input_buf.clone()),
                WriteDescriptorSet::buffer(1, output_buf.clone()),
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

        let push_constants: [u32; 3] = [op_type, alpha.to_bits(), total_elements as u32];

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
                .push_constants(pipeline.layout().clone(), 0, push_constants)
                .unwrap()
                .dispatch([((total_elements + 255) / 256) as u32, 1, 1])
                .unwrap();
        }

        let command_buffer = builder.build().expect("build command buffer");
        let future = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let staging_size = total_elements as u64 * std::mem::size_of::<f32>() as u64;
        let staging_buf: Subbuffer<[f32]> = Buffer::new_unsized(
            self.context.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            staging_size,
        )
        .expect("staging buffer");

        let mut copy_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        copy_builder
            .copy_buffer(CopyBufferInfo::buffers(output_buf, staging_buf.clone()))
            .unwrap();
        let copy_cb = copy_builder.build().unwrap();
        let future2 = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), copy_cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future2.wait(None).unwrap();

        let data = staging_buf.read().expect("read staging buffer");
        Mat::from_fn(batch, features, |r, c| data[r * features + c])
    }

    pub fn run_relu_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        self.run_activation_forward(input, 0, 0.0)
    }
    pub fn run_sigmoid_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        self.run_activation_forward(input, 1, 0.0)
    }
    pub fn run_tanh_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        self.run_activation_forward(input, 2, 0.0)
    }
    pub fn run_leaky_relu_forward(&self, input: &Mat<f32>, alpha: f32) -> Mat<f32> {
        self.run_activation_forward(input, 3, alpha)
    }

    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    fn create_storage_buffer_from_slice(
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
}