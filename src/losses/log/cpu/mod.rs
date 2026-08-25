// src/losses/log/cpu/mod.rs

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::losses::BufferedElemCube;
use crate::losses::log::Log;

impl BufferedElemCube for Log {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("Log forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("Log forward: expected CPU buffer");

        debug_assert_eq!(src.len(), dst.len());

        for (o, &x) in dst.iter_mut().zip(src.iter()) {
            *o = x.ln();
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
        let x = x_guard.as_slice().expect("Log backward: expected CPU buffer");

        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("Log backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("Log backward: expected CPU buffer");

        debug_assert_eq!(x.len(), go.len());
        debug_assert_eq!(x.len(), gi.len());

        for i in 0..x.len() {
            gi[i] = go[i] / x[i];
        }
    }
}
