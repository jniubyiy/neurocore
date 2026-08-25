// src/losses/sub/cpu/mod.rs

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::losses::BufferedElemCube;
use crate::losses::sub::Sub;

impl BufferedElemCube for Sub {
    fn in_features(&self) -> usize {
        2 * self.features
    }

    fn out_features(&self) -> usize {
        self.features
    }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let rows = input.rows();
        let f = self.features;

        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Sub forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("Sub forward: expected CPU buffer");

        debug_assert_eq!(src.len(), rows * 2 * f);
        debug_assert_eq!(dst.len(), rows * f);

        for r in 0..rows {
            for c in 0..f {
                dst[c * rows + r] = src[c * rows + r] - src[(c + f) * rows + r];
            }
        }
    }

    fn backward_buffered(
        &self,
        _input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        let rows = grad_out.rows();
        let f = self.features;

        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("Sub backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("Sub backward: expected CPU buffer");

        debug_assert_eq!(go.len(), rows * f);
        debug_assert_eq!(gi.len(), rows * 2 * f);

        for r in 0..rows {
            for c in 0..f {
                let g = go[c * rows + r];
                gi[c * rows + r] = g;
                gi[(c + f) * rows + r] = -g;
            }
        }
    }
}
