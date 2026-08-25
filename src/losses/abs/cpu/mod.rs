// src/losses/abs/cpu/mod.rs

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::losses::BufferedElemCube;
use crate::losses::abs::Abs;

impl BufferedElemCube for Abs {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Abs forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("Abs forward: expected CPU buffer");

        debug_assert_eq!(src.len(), dst.len());

        for (o, &x) in dst.iter_mut().zip(src.iter()) {
            *o = x.abs();
        }
    }

    fn backward_buffered(
        &self,
        input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        let x_guard = input.read();
        let x = x_guard.as_slice().expect("Abs backward: expected CPU buffer");

        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("Abs backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("Abs backward: expected CPU buffer");

        debug_assert_eq!(x.len(), go.len());
        debug_assert_eq!(x.len(), gi.len());

        for i in 0..x.len() {
            if x[i] > 0.0 {
                gi[i] = go[i];
            } else if x[i] < 0.0 {
                gi[i] = -go[i];
            } else {
                gi[i] = 0.0;
            }
        }
    }
}
