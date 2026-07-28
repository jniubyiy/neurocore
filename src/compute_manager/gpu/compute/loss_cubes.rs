// src/compute_manager/gpu/compute/loss_cubes.rs

use faer::Mat;
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};
use super::base::GpuCompute;

impl GpuCompute {
    // --- Sub ---
    pub fn run_sub_forward(&self, pred: &Mat<f32>, target: &Mat<f32>) -> Mat<f32> {
        let total = pred.nrows() * pred.ncols();
        let (a_buf, a_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(pred));
        let (b_buf, b_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(target));
        let (out_buf, out_id) = self.run_elementwise_2in_1out(
            self.pipeline_cache.sub_fwd.clone(),
            a_buf, b_buf, total,
            [total as u32],
        );
        let mat = self.read_buffer_to_mat(out_buf, out_id, pred.nrows(), pred.ncols());
        self.release_buffer(a_id);
        self.release_buffer(b_id);
        mat
    }

    pub fn run_sub_backward(&self, grad_out: &Mat<f32>) -> (Mat<f32>, Mat<f32>) {
        let total = grad_out.nrows() * grad_out.ncols();
        let (go_buf, go_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(grad_out));
        let (ga_buf, ga_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);
        let (gb_buf, gb_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let set_layout = self.pipeline_cache.sub_bwd.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, go_buf.clone()),
                WriteDescriptorSet::buffer(1, ga_buf.clone()),
                WriteDescriptorSet::buffer(2, gb_buf.clone()),
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

        let dispatch_dim = [((total + 255) / 256) as u32, 1, 1];
        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline_cache.sub_bwd.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline_cache.sub_bwd.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(self.pipeline_cache.sub_bwd.layout().clone(), 0, [total as u32])
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

        let ga = self.read_buffer_to_mat(ga_buf, ga_id, grad_out.nrows(), grad_out.ncols());
        let gb = self.read_buffer_to_mat(gb_buf, gb_id, grad_out.nrows(), grad_out.ncols());
        self.release_buffer(go_id);
        (ga, gb)
    }

    // --- Square ---
    pub fn run_square_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (out_buf, out_id) = self.run_elementwise_1in_1out(
            self.pipeline_cache.square_fwd.clone(),
            in_buf, total,
            [total as u32],
        );
        let mat = self.read_buffer_to_mat(out_buf, out_id, input.nrows(), input.ncols());
        self.release_buffer(in_id);
        mat
    }

    pub fn run_square_backward(&self, input: &Mat<f32>, grad_out: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (go_buf, go_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let set_layout = self.pipeline_cache.square_bwd.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, go_buf.clone()),
                WriteDescriptorSet::buffer(2, gi_buf.clone()),
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

        let dispatch_dim = [((total + 255) / 256) as u32, 1, 1];
        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline_cache.square_bwd.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline_cache.square_bwd.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(self.pipeline_cache.square_bwd.layout().clone(), 0, [total as u32])
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

        let mat = self.read_buffer_to_mat(gi_buf, gi_id, input.nrows(), input.ncols());
        self.release_buffer(in_id);
        self.release_buffer(go_id);
        mat
    }

    // --- Abs ---
    pub fn run_abs_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (out_buf, out_id) = self.run_elementwise_1in_1out(
            self.pipeline_cache.abs_fwd.clone(),
            in_buf, total,
            [total as u32],
        );
        let mat = self.read_buffer_to_mat(out_buf, out_id, input.nrows(), input.ncols());
        self.release_buffer(in_id);
        mat
    }

    pub fn run_abs_backward(&self, input: &Mat<f32>, grad_out: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (go_buf, go_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let set_layout = self.pipeline_cache.abs_bwd.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, go_buf.clone()),
                WriteDescriptorSet::buffer(2, gi_buf.clone()),
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

        let dispatch_dim = [((total + 255) / 256) as u32, 1, 1];
        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline_cache.abs_bwd.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline_cache.abs_bwd.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(self.pipeline_cache.abs_bwd.layout().clone(), 0, [total as u32])
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

        let mat = self.read_buffer_to_mat(gi_buf, gi_id, input.nrows(), input.ncols());
        self.release_buffer(in_id);
        self.release_buffer(go_id);
        mat
    }

    // --- Log1p ---
    pub fn run_log1p_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (out_buf, out_id) = self.run_elementwise_1in_1out(
            self.pipeline_cache.log1p_fwd.clone(),
            in_buf, total,
            [total as u32],
        );
        let mat = self.read_buffer_to_mat(out_buf, out_id, input.nrows(), input.ncols());
        self.release_buffer(in_id);
        mat
    }

    pub fn run_log1p_backward(&self, input: &Mat<f32>, grad_out: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (go_buf, go_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let set_layout = self.pipeline_cache.log1p_bwd.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, go_buf.clone()),
                WriteDescriptorSet::buffer(2, gi_buf.clone()),
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

        let dispatch_dim = [((total + 255) / 256) as u32, 1, 1];
        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline_cache.log1p_bwd.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline_cache.log1p_bwd.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(self.pipeline_cache.log1p_bwd.layout().clone(), 0, [total as u32])
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

        let mat = self.read_buffer_to_mat(gi_buf, gi_id, input.nrows(), input.ncols());
        self.release_buffer(in_id);
        self.release_buffer(go_id);
        mat
    }

    // --- AbsDiff ---
    pub fn run_absdiff_forward(&self, a: &Mat<f32>, b: &Mat<f32>) -> Mat<f32> {
        let total = a.nrows() * a.ncols();
        let (a_buf, a_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(a));
        let (b_buf, b_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(b));
        let (out_buf, out_id) = self.run_elementwise_2in_1out(
            self.pipeline_cache.absdiff_fwd.clone(),
            a_buf, b_buf, total,
            [total as u32],
        );
        let mat = self.read_buffer_to_mat(out_buf, out_id, a.nrows(), a.ncols());
        self.release_buffer(a_id);
        self.release_buffer(b_id);
        mat
    }

    pub fn run_absdiff_backward(&self, a: &Mat<f32>, b: &Mat<f32>, grad_out: &Mat<f32>) -> (Mat<f32>, Mat<f32>) {
        let total = a.nrows() * a.ncols();
        let (a_buf, a_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(a));
        let (b_buf, b_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(b));
        let (go_buf, go_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(grad_out));
        let (ga_buf, ga_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);
        let (gb_buf, gb_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let set_layout = self.pipeline_cache.absdiff_bwd.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, a_buf.clone()),
                WriteDescriptorSet::buffer(1, b_buf.clone()),
                WriteDescriptorSet::buffer(2, go_buf.clone()),
                WriteDescriptorSet::buffer(3, ga_buf.clone()),
                WriteDescriptorSet::buffer(4, gb_buf.clone()),
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

        let dispatch_dim = [((total + 255) / 256) as u32, 1, 1];
        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline_cache.absdiff_bwd.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline_cache.absdiff_bwd.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(self.pipeline_cache.absdiff_bwd.layout().clone(), 0, [total as u32])
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

        let ga = self.read_buffer_to_mat(ga_buf, ga_id, a.nrows(), a.ncols());
        let gb = self.read_buffer_to_mat(gb_buf, gb_id, a.nrows(), a.ncols());
        self.release_buffer(a_id);
        self.release_buffer(b_id);
        self.release_buffer(go_id);
        (ga, gb)
    }

    // --- Log ---
    pub fn run_log_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (out_buf, out_id) = self.run_elementwise_1in_1out(
            self.pipeline_cache.log_fwd.clone(),
            in_buf, total,
            [total as u32],
        );
        let mat = self.read_buffer_to_mat(out_buf, out_id, input.nrows(), input.ncols());
        self.release_buffer(in_id);
        mat
    }

    pub fn run_log_backward(&self, input: &Mat<f32>, grad_out: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (go_buf, go_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let set_layout = self.pipeline_cache.log_bwd.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, go_buf.clone()),
                WriteDescriptorSet::buffer(2, gi_buf.clone()),
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

        let dispatch_dim = [((total + 255) / 256) as u32, 1, 1];
        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline_cache.log_bwd.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline_cache.log_bwd.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(self.pipeline_cache.log_bwd.layout().clone(), 0, [total as u32])
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

        let mat = self.read_buffer_to_mat(gi_buf, gi_id, input.nrows(), input.ncols());
        self.release_buffer(in_id);
        self.release_buffer(go_id);
        mat
    }

    // --- Neg ---
    pub fn run_neg_forward(&self, input: &Mat<f32>) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (out_buf, out_id) = self.run_elementwise_1in_1out(
            self.pipeline_cache.neg_fwd.clone(),
            in_buf, total,
            [total as u32],
        );
        let mat = self.read_buffer_to_mat(out_buf, out_id, input.nrows(), input.ncols());
        self.release_buffer(in_id);
        mat
    }

    pub fn run_neg_backward(&self, grad_out: &Mat<f32>) -> Mat<f32> {
        let total = grad_out.nrows() * grad_out.ncols();
        let (go_buf, go_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let set_layout = self.pipeline_cache.neg_bwd.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, go_buf.clone()),
                WriteDescriptorSet::buffer(1, gi_buf.clone()),
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

        let dispatch_dim = [((total + 255) / 256) as u32, 1, 1];
        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline_cache.neg_bwd.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline_cache.neg_bwd.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(self.pipeline_cache.neg_bwd.layout().clone(), 0, [total as u32])
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

        let mat = self.read_buffer_to_mat(gi_buf, gi_id, grad_out.nrows(), grad_out.ncols());
        self.release_buffer(go_id);
        mat
    }

    // --- Mul ---
    pub fn run_mul_forward(&self, a: &Mat<f32>, b: &Mat<f32>) -> Mat<f32> {
        let total = a.nrows() * a.ncols();
        let (a_buf, a_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(a));
        let (b_buf, b_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(b));
        let (out_buf, out_id) = self.run_elementwise_2in_1out(
            self.pipeline_cache.mul_fwd.clone(),
            a_buf, b_buf, total,
            [total as u32],
        );
        let mat = self.read_buffer_to_mat(out_buf, out_id, a.nrows(), a.ncols());
        self.release_buffer(a_id);
        self.release_buffer(b_id);
        mat
    }

    pub fn run_mul_backward(&self, a: &Mat<f32>, b: &Mat<f32>, grad_out: &Mat<f32>) -> (Mat<f32>, Mat<f32>) {
        let total = a.nrows() * a.ncols();
        let (a_buf, a_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(a));
        let (b_buf, b_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(b));
        let (go_buf, go_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(grad_out));
        let (ga_buf, ga_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);
        let (gb_buf, gb_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let set_layout = self.pipeline_cache.mul_bwd.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, a_buf.clone()),
                WriteDescriptorSet::buffer(1, b_buf.clone()),
                WriteDescriptorSet::buffer(2, go_buf.clone()),
                WriteDescriptorSet::buffer(3, ga_buf.clone()),
                WriteDescriptorSet::buffer(4, gb_buf.clone()),
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

        let dispatch_dim = [((total + 255) / 256) as u32, 1, 1];
        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline_cache.mul_bwd.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline_cache.mul_bwd.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(self.pipeline_cache.mul_bwd.layout().clone(), 0, [total as u32])
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

        let ga = self.read_buffer_to_mat(ga_buf, ga_id, a.nrows(), a.ncols());
        let gb = self.read_buffer_to_mat(gb_buf, gb_id, a.nrows(), a.ncols());
        self.release_buffer(a_id);
        self.release_buffer(b_id);
        self.release_buffer(go_id);
        (ga, gb)
    }

    // --- AddScalar ---
    pub fn run_addscalar_forward(&self, input: &Mat<f32>, scalar: f32) -> Mat<f32> {
        let total = input.nrows() * input.ncols();
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let push_data = [total as u32, scalar.to_bits()];
        let (out_buf, out_id) = self.run_elementwise_1in_1out(
            self.pipeline_cache.addscalar_fwd.clone(),
            in_buf, total,
            push_data,
        );
        let mat = self.read_buffer_to_mat(out_buf, out_id, input.nrows(), input.ncols());
        self.release_buffer(in_id);
        mat
    }

    pub fn run_addscalar_backward(&self, grad_out: &Mat<f32>) -> Mat<f32> {
        let total = grad_out.nrows() * grad_out.ncols();
        let (go_buf, go_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let set_layout = self.pipeline_cache.addscalar_bwd.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, go_buf.clone()),
                WriteDescriptorSet::buffer(1, gi_buf.clone()),
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

        let dispatch_dim = [((total + 255) / 256) as u32, 1, 1];
        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline_cache.addscalar_bwd.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline_cache.addscalar_bwd.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(self.pipeline_cache.addscalar_bwd.layout().clone(), 0, [total as u32])
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

        let mat = self.read_buffer_to_mat(gi_buf, gi_id, grad_out.nrows(), grad_out.ncols());
        self.release_buffer(go_id);
        mat
    }

    // --- CrossEntropy ---
    pub fn run_cross_entropy_forward(&self, logits_and_target: &Mat<f32>, num_classes: usize) -> Mat<f32> {
        let batch = logits_and_target.nrows();
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(logits_and_target));
        let (out_buf, out_id) = self.create_buffer(batch, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let set_layout = self.pipeline_cache.cross_entropy_fwd.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
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

        let dispatch_dim = [batch as u32, 1, 1];
        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline_cache.cross_entropy_fwd.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline_cache.cross_entropy_fwd.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(self.pipeline_cache.cross_entropy_fwd.layout().clone(), 0, [batch as u32, num_classes as u32])
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

        let mat = self.read_buffer_to_mat(out_buf, out_id, batch, 1);
        self.release_buffer(in_id);
        mat
    }

    pub fn run_cross_entropy_backward(
        &self,
        logits_and_target: &Mat<f32>,
        grad_out: &Mat<f32>,
        num_classes: usize,
    ) -> Mat<f32> {
        let batch = logits_and_target.nrows();
        let total_elements = batch * (num_classes + 1);
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(logits_and_target));
        let (go_buf, go_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_id) = self.create_buffer(total_elements, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let set_layout = self.pipeline_cache.cross_entropy_bwd.layout().set_layouts().get(0).unwrap().clone();
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, go_buf.clone()),
                WriteDescriptorSet::buffer(2, gi_buf.clone()),
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

        let dispatch_dim = [batch as u32, 1, 1];
        unsafe {
            builder
                .bind_pipeline_compute(self.pipeline_cache.cross_entropy_bwd.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline_cache.cross_entropy_bwd.layout().clone(),
                    0,
                    descriptor_set,
                )
                .unwrap()
                .push_constants(self.pipeline_cache.cross_entropy_bwd.layout().clone(), 0, [batch as u32, num_classes as u32])
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

        let mat = self.read_buffer_to_mat(gi_buf, gi_id, batch, num_classes + 1);
        self.release_buffer(in_id);
        self.release_buffer(go_id);
        mat
    }
}