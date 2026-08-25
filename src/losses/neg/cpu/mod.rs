// src/losses/neg/cpu/mod.rs

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::losses::BufferedElemCube;
use crate::losses::neg::Neg;

impl BufferedElemCube for Neg {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Neg forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("Neg forward: expected CPU buffer");

        debug_assert_eq!(src.len(), dst.len());

        for (o, &x) in dst.iter_mut().zip(src.iter()) {
            *o = -x;
        }
    }

    fn backward_buffered(
        &self,
        _input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("Neg backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("Neg backward: expected CPU buffer");

        debug_assert_eq!(go.len(), gi.len());

        for (o, &g) in gi.iter_mut().zip(go.iter()) {
            *o = -g;
        }
    }
}
