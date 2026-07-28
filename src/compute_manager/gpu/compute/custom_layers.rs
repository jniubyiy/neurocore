// src/compute_manager/gpu/compute/custom_layers.rs

use std::sync::Arc;
use faer::Mat;
use vulkano::buffer::{Subbuffer, BufferUsage};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};
use super::base::GpuCompute;

impl GpuCompute {
    // ---------- Memory ----------
    pub fn init_memory_state(&mut self, features: usize, _alpha: f32) {
        let mut state = Vec::with_capacity(2 * features);
        state.resize(features, f32::MAX);
        state.resize(2 * features, f32::MIN);
        let (buf, _id) = self.create_storage_buffer_from_slice(&state);
        // Игнорируем id – буфер будет жить, пока хранится в self.memory_state.
        // Освобождение произойдёт при дропе GpuCompute (или никогда, если не нужно).
        self.memory_state = Some(buf);
    }

    pub fn run_memory_forward(
        &self,
        input: &Mat<f32>,
        alpha: f32,
        state: &Subbuffer<[f32]>,
    ) -> Mat<f32> {
        let batch = input.nrows();
        let features = input.ncols();
        let total = batch * features;
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (out_buf, out_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let pipeline = self.pipeline_cache.memory_fwd.clone();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let push = [batch as u32, features as u32, alpha.to_bits()];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, state.clone()),
                WriteDescriptorSet::buffer(2, out_buf.clone()),
            ],
            [],
        )
        .expect("memory_fwd descriptor set");

        self.run_custom_shader(pipeline, descriptor_set, push, total);

        let mat = self.read_buffer_to_mat(out_buf, out_id, batch, features);
        self.release_buffer(in_id);
        mat
    }

    pub fn run_memory_backward(
        &self,
        grad_out: &Mat<f32>,
        alpha: f32,
    ) -> Mat<f32> {
        let total = grad_out.nrows() * grad_out.ncols();
        let (go_buf, go_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(grad_out));
        let (gi_buf, gi_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let pipeline = self.pipeline_cache.memory_bwd.clone();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let push = [total as u32, alpha.to_bits()];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, go_buf.clone()),
                WriteDescriptorSet::buffer(1, gi_buf.clone()),
            ],
            [],
        )
        .expect("memory_bwd descriptor set");

        self.run_custom_shader(pipeline, descriptor_set, push, total);

        let mat = self.read_buffer_to_mat(gi_buf, gi_id, grad_out.nrows(), grad_out.ncols());
        self.release_buffer(go_id);
        mat
    }

    // ---------- SoftSparseGate ----------
    pub fn run_softsparse_forward(
        &self,
        input: &Mat<f32>,
        thresholds: &[f32],
        temperature: f32,
    ) -> Mat<f32> {
        let batch = input.nrows();
        let features = input.ncols();
        let total = batch * features;
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (thresh_buf, th_id) = self.create_storage_buffer_from_slice(thresholds);
        let (out_buf, out_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let pipeline = self.pipeline_cache.softsparse_fwd.clone();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let push = [total as u32, temperature.to_bits(), features as u32];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, thresh_buf.clone()),
                WriteDescriptorSet::buffer(2, out_buf.clone()),
            ],
            [],
        )
        .expect("softsparse_fwd descriptor set");

        self.run_custom_shader(pipeline, descriptor_set, push, total);

        let mat = self.read_buffer_to_mat(out_buf, out_id, batch, features);
        self.release_buffer(in_id);
        self.release_buffer(th_id);
        mat
    }

    pub fn run_softsparse_backward(
        &self,
        input: &Mat<f32>,
        grad_out: &Mat<f32>,
        thresholds: &[f32],
        temperature: f32,
    ) -> (Mat<f32>, Vec<f32>) {
        let batch = input.nrows();
        let features = input.ncols();
        let total = batch * features;
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (go_buf, go_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(grad_out));
        let (thresh_buf, th_id) = self.create_storage_buffer_from_slice(thresholds);
        let (gi_buf, gi_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);
        let (gthresh_buf, gth_id) = self.create_buffer(features, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let pipeline = self.pipeline_cache.softsparse_bwd.clone();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let push = [total as u32, temperature.to_bits(), features as u32];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, go_buf.clone()),
                WriteDescriptorSet::buffer(2, thresh_buf.clone()),
                WriteDescriptorSet::buffer(3, gi_buf.clone()),
                WriteDescriptorSet::buffer(4, gthresh_buf.clone()),
            ],
            [],
        )
        .expect("softsparse_bwd descriptor set");

        self.run_custom_shader(pipeline, descriptor_set, push, total);

        let gi = self.read_buffer_to_mat(gi_buf, gi_id, batch, features);
        let (staging, staging_id) = self.create_buffer(features, BufferUsage::TRANSFER_DST);
        self.copy_buffer_sync(gthresh_buf, staging.clone());
        let gthresh = {
            let guard = staging.read().unwrap();
            guard[..features].to_vec()
        };
        self.release_buffer(staging_id);
        self.release_buffer(in_id);
        self.release_buffer(go_id);
        self.release_buffer(th_id);
        self.release_buffer(gth_id);
        (gi, gthresh)
    }

    // ---------- SoftKeepGate ----------
    pub fn run_softkeep_forward(
        &self,
        input: &Mat<f32>,
        thresholds: &[f32],
        temperature: f32,
    ) -> Mat<f32> {
        let batch = input.nrows();
        let features = input.ncols();
        let total = batch * features;
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (thresh_buf, th_id) = self.create_storage_buffer_from_slice(thresholds);
        let (out_buf, out_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let pipeline = self.pipeline_cache.softkeep_fwd.clone();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let push = [total as u32, temperature.to_bits(), features as u32];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, thresh_buf.clone()),
                WriteDescriptorSet::buffer(2, out_buf.clone()),
            ],
            [],
        )
        .expect("softkeep_fwd descriptor set");

        self.run_custom_shader(pipeline, descriptor_set, push, total);

        let mat = self.read_buffer_to_mat(out_buf, out_id, batch, features);
        self.release_buffer(in_id);
        self.release_buffer(th_id);
        mat
    }

    pub fn run_softkeep_backward(
        &self,
        input: &Mat<f32>,
        grad_out: &Mat<f32>,
        thresholds: &[f32],
        temperature: f32,
    ) -> (Mat<f32>, Vec<f32>) {
        let batch = input.nrows();
        let features = input.ncols();
        let total = batch * features;
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (go_buf, go_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(grad_out));
        let (thresh_buf, th_id) = self.create_storage_buffer_from_slice(thresholds);
        let (gi_buf, gi_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);
        let (gthresh_buf, gth_id) = self.create_buffer(features, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let pipeline = self.pipeline_cache.softkeep_bwd.clone();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let push = [total as u32, temperature.to_bits(), features as u32];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, go_buf.clone()),
                WriteDescriptorSet::buffer(2, thresh_buf.clone()),
                WriteDescriptorSet::buffer(3, gi_buf.clone()),
                WriteDescriptorSet::buffer(4, gthresh_buf.clone()),
            ],
            [],
        )
        .expect("softkeep_bwd descriptor set");

        self.run_custom_shader(pipeline, descriptor_set, push, total);

        let gi = self.read_buffer_to_mat(gi_buf, gi_id, batch, features);
        let (staging, staging_id) = self.create_buffer(features, BufferUsage::TRANSFER_DST);
        self.copy_buffer_sync(gthresh_buf, staging.clone());
        let gthresh = {
            let guard = staging.read().unwrap();
            guard[..features].to_vec()
        };
        self.release_buffer(staging_id);
        self.release_buffer(in_id);
        self.release_buffer(go_id);
        self.release_buffer(th_id);
        self.release_buffer(gth_id);
        (gi, gthresh)
    }

    // ---------- DualAnchor ----------
    pub fn run_dualanchor_forward(
        &self,
        input: &Mat<f32>,
        min_vals: &[f32],
        max_vals: &[f32],
        alpha: f32,
    ) -> Mat<f32> {
        let batch = input.nrows();
        let features = input.ncols();
        let total = batch * features;
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (min_buf, min_id) = self.create_storage_buffer_from_slice(min_vals);
        let (max_buf, max_id) = self.create_storage_buffer_from_slice(max_vals);
        let (out_buf, out_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let pipeline = self.pipeline_cache.dualanchor_fwd.clone();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let push = [total as u32, features as u32, alpha.to_bits()];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, min_buf.clone()),
                WriteDescriptorSet::buffer(2, max_buf.clone()),
                WriteDescriptorSet::buffer(3, out_buf.clone()),
            ],
            [],
        )
        .expect("dualanchor_fwd descriptor set");

        self.run_custom_shader(pipeline, descriptor_set, push, total);

        let mat = self.read_buffer_to_mat(out_buf, out_id, batch, features);
        self.release_buffer(in_id);
        self.release_buffer(min_id);
        self.release_buffer(max_id);
        mat
    }

    pub fn run_dualanchor_backward(
        &self,
        input: &Mat<f32>,
        grad_out: &Mat<f32>,
        min_vals: &[f32],
        max_vals: &[f32],
        alpha: f32,
    ) -> (Mat<f32>, Vec<f32>) {
        let batch = input.nrows();
        let features = input.ncols();
        let total = batch * features;
        let (in_buf, in_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(input));
        let (go_buf, go_id) = self.create_storage_buffer_from_slice(&Self::mat_to_flat(grad_out));
        let (min_buf, min_id) = self.create_storage_buffer_from_slice(min_vals);
        let (max_buf, max_id) = self.create_storage_buffer_from_slice(max_vals);
        let (gi_buf, gi_id) = self.create_buffer(total, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);
        let (gmin_buf, gmin_id) = self.create_buffer(features, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);
        let (gmax_buf, gmax_id) = self.create_buffer(features, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);
        let (galpha_buf, ga_id) = self.create_buffer(1, BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC);

        let pipeline = self.pipeline_cache.dualanchor_bwd.clone();
        let set_layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
        let push = [total as u32, features as u32, alpha.to_bits()];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, in_buf.clone()),
                WriteDescriptorSet::buffer(1, go_buf.clone()),
                WriteDescriptorSet::buffer(2, min_buf.clone()),
                WriteDescriptorSet::buffer(3, max_buf.clone()),
                WriteDescriptorSet::buffer(4, gi_buf.clone()),
                WriteDescriptorSet::buffer(5, gmin_buf.clone()),
                WriteDescriptorSet::buffer(6, gmax_buf.clone()),
                WriteDescriptorSet::buffer(7, galpha_buf.clone()),
            ],
            [],
        )
        .expect("dualanchor_bwd descriptor set");

        self.run_custom_shader(pipeline, descriptor_set, push, total);

        let gi = self.read_buffer_to_mat(gi_buf, gi_id, batch, features);

        let (gmin_staging, gms_id) = self.create_buffer(features, BufferUsage::TRANSFER_DST);
        self.copy_buffer_sync(gmin_buf, gmin_staging.clone());
        let gmin = {
            let guard = gmin_staging.read().unwrap();
            guard[..features].to_vec()
        };
        self.release_buffer(gms_id);

        let (gmax_staging, gxs_id) = self.create_buffer(features, BufferUsage::TRANSFER_DST);
        self.copy_buffer_sync(gmax_buf, gmax_staging.clone());
        let gmax = {
            let guard = gmax_staging.read().unwrap();
            guard[..features].to_vec()
        };
        self.release_buffer(gxs_id);

        let (galpha_staging, gas_id) = self.create_buffer(1, BufferUsage::TRANSFER_DST);
        self.copy_buffer_sync(galpha_buf, galpha_staging.clone());
        let galpha = {
            let guard = galpha_staging.read().unwrap();
            guard[0]
        };
        self.release_buffer(gas_id);

        let mut grad = Vec::with_capacity(2 * features + 1);
        grad.extend_from_slice(&gmin);
        grad.extend_from_slice(&gmax);
        grad.push(galpha);

        self.release_buffer(in_id);
        self.release_buffer(go_id);
        self.release_buffer(min_id);
        self.release_buffer(max_id);
        self.release_buffer(gmin_id);
        self.release_buffer(gmax_id);
        self.release_buffer(ga_id);

        (gi, grad)
    }

    // ---------- универсальный запуск шейдера ----------
    fn run_custom_shader<const N: usize>(
        &self,
        pipeline: Arc<vulkano::pipeline::ComputePipeline>,
        descriptor_set: Arc<DescriptorSet>,
        push: [u32; N],
        total_elements: usize,
    ) {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("custom shader builder");

        let dispatch_dim = [((total_elements + 255) / 256) as u32, 1, 1];

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
                .push_constants(pipeline.layout().clone(), 0, push)
                .unwrap()
                .dispatch(dispatch_dim)
                .unwrap();
        }

        let command_buffer = builder.build().expect("build custom command buffer");
        let future = sync::now(self.context.device.clone())
            .then_execute(self.context.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
    }
}