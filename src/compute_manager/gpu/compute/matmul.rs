// src/compute_manager/gpu/compute/matmul.rs

use faer::Mat;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};
use super::base::GpuCompute;

impl GpuCompute {
    fn run_mat_mul_internal(
        &self,
        a: &Mat<f32>,
        b: &Mat<f32>,
        output_rows: usize,
        output_cols: usize,
    ) -> Subbuffer<[f32]> {
        let k = a.ncols();
        assert_eq!(k, b.nrows());
        let m = output_rows;
        let n = output_cols;

        let a_data: Vec<f32> = (0..a.nrows())
            .flat_map(|r| (0..a.ncols()).map(move |c| a[(r, c)]))
            .collect();
        let b_data: Vec<f32> = (0..b.nrows())
            .flat_map(|r| (0..b.ncols()).map(move |c| b[(r, c)]))
            .collect();

        let a_buf = Self::create_storage_buffer_from_slice(&self.context.memory_allocator, &a_data);
        let b_buf = Self::create_storage_buffer_from_slice(&self.context.memory_allocator, &b_data);
        let out_buf = self.create_buffer(m * n, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let pipeline = self.pipeline_cache.mat_mul_pipeline();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, a_buf.clone()),
                WriteDescriptorSet::buffer(1, b_buf.clone()),
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

        let dispatch_dim = [
            ((m + 15) / 16) as u32,
            ((n + 15) / 16) as u32,
            1u32,
        ];
        let push_constants: [u32; 3] = [m as u32, n as u32, k as u32];

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

        out_buf
    }

    pub fn run_mat_mul(&self, a: &Mat<f32>, b: &Mat<f32>) -> Mat<f32> {
        let out_buf = self.run_mat_mul_internal(a, b, a.nrows(), b.ncols());
        self.read_buffer_to_mat(out_buf, a.nrows(), b.ncols())
    }

    pub fn run_reduce_sum_cols(&self, mat: &Mat<f32>) -> Vec<f32> {
        let rows = mat.nrows();
        let cols = mat.ncols();

        let input_buf = Self::create_storage_buffer_from_slice(
            &self.context.memory_allocator,
            &(0..rows).flat_map(|r| (0..cols).map(move |c| mat[(r, c)])).collect::<Vec<f32>>(),
        );
        let output_buf = self.create_buffer(cols, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let pipeline = self.pipeline_cache.reduce_pipeline();
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

        let dispatch_dim = [cols as u32, 1, 1];
        let push_constants: [u32; 1] = [rows as u32];

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

        let staging = self.create_buffer(cols, BufferUsage::TRANSFER_DST);
        self.copy_buffer_sync(output_buf, staging.clone());
        let data = staging.read().unwrap();
        data[..cols].to_vec()
    }
}