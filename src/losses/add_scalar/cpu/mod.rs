// src/losses/add_scalar/cpu/mod.rs

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::losses::BufferedElemCube;
use crate::losses::add_scalar::AddScalar;

impl BufferedElemCube for AddScalar {
    fn in_features(&self) -> usize { 1 }
    fn out_features(&self) -> usize { 1 }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let scalar = self.0;
        let src_guard = input.read();
        let src = src_guard.as_slice().expect("AddScalar forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("AddScalar forward: expected CPU buffer");

        debug_assert_eq!(src.len(), dst.len());

        for (o, &x) in dst.iter_mut().zip(src.iter()) {
            *o = x + scalar;
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
        let go = go_guard.as_slice().expect("AddScalar backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("AddScalar backward: expected CPU buffer");

        debug_assert_eq!(go.len(), gi.len());
        gi.copy_from_slice(go);
    }
}
