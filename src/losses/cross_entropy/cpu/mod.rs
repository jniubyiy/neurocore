// src/losses/cross_entropy/cpu/mod.rs

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::losses::BufferedElemCube;
use crate::losses::cross_entropy::CrossEntropyWithLogits;

impl BufferedElemCube for CrossEntropyWithLogits {
    fn in_features(&self) -> usize {
        self.num_classes + 1
    }

    fn out_features(&self) -> usize {
        1
    }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        assert!(!input.is_gpu() && !output.is_gpu(),
            "BufferedElemCube for CrossEntropyWithLogits supports only CPU buffers");

        let batch = input.rows();
        let nclass = self.num_classes;

        let src_guard = input.read();
        let src = src_guard.as_slice().expect("CrossEntropy forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("CrossEntropy forward: expected CPU buffer");

        debug_assert_eq!(src.len(), batch * (nclass + 1));
        debug_assert_eq!(dst.len(), batch);

        for r in 0..batch {
            let class_idx = src[nclass * batch + r] as usize;

            let mut max_val = f32::NEG_INFINITY;
            for c in 0..nclass {
                max_val = max_val.max(src[c * batch + r]);
            }

            let mut exp_sum = 0.0f32;
            for c in 0..nclass {
                exp_sum += (src[c * batch + r] - max_val).exp();
            }

            dst[r] = -src[class_idx * batch + r] + max_val + exp_sum.ln();
        }
    }

    fn backward_buffered(
        &self,
        input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        assert!(!input.is_gpu() && !grad_out.is_gpu() && !grad_in.is_gpu(),
            "BufferedElemCube for CrossEntropyWithLogits supports only CPU buffers");

        let batch = input.rows();
        let nclass = self.num_classes;

        let src_guard = input.read();
        let src = src_guard.as_slice().expect("CrossEntropy backward: expected CPU buffer");
        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("CrossEntropy backward: expected CPU buffer");
        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("CrossEntropy backward: expected CPU buffer");

        debug_assert_eq!(src.len(), batch * (nclass + 1));
        debug_assert_eq!(go.len(), batch);
        debug_assert_eq!(gi.len(), batch * (nclass + 1));

        for r in 0..batch {
            let class_idx = src[nclass * batch + r] as usize;
            let g = go[r];

            let mut max_val = f32::NEG_INFINITY;
            for c in 0..nclass {
                max_val = max_val.max(src[c * batch + r]);
            }

            let mut exp_sum = 0.0f32;
            for c in 0..nclass {
                exp_sum += (src[c * batch + r] - max_val).exp();
            }

            for c in 0..nclass {
                let softmax_c = (src[c * batch + r] - max_val).exp() / exp_sum;
                let indicator = if c == class_idx { 1.0 } else { 0.0 };
                gi[c * batch + r] = g * (softmax_c - indicator);
            }
            gi[nclass * batch + r] = 0.0;
        }
    }
}

