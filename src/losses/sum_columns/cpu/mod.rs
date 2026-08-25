// src/losses/sum_columns/cpu/mod.rs

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::losses::BufferedElemCube;
use crate::losses::sum_columns::SumColumns;

impl BufferedElemCube for SumColumns {
    fn in_features(&self) -> usize { 0 }
    fn out_features(&self) -> usize { 1 }

    fn forward_buffered(&self, input: &MatrixBufferHandle, output: &mut MatrixBufferHandle) {
        let rows = input.rows();
        let cols = input.cols();

        let src_guard = input.read();
        let src = src_guard.as_slice().expect("SumColumns forward: expected CPU buffer");

        let mut dst_guard = output.write();
        let dst = dst_guard.as_slice_mut().expect("SumColumns forward: expected CPU buffer");

        debug_assert_eq!(src.len(), rows * cols);
        debug_assert_eq!(dst.len(), rows);

        for r in 0..rows {
            let mut sum = 0.0;
            for c in 0..cols {
                sum += src[c * rows + r];
            }
            dst[r] = sum;
        }
    }

    fn backward_buffered(
        &self,
        input: &MatrixBufferHandle,
        _output_cache: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        grad_in: &mut MatrixBufferHandle,
    ) {
        let rows = grad_out.rows();
        let cols = input.cols();

        let go_guard = grad_out.read();
        let go = go_guard.as_slice().expect("SumColumns backward: expected CPU buffer");

        let mut gi_guard = grad_in.write();
        let gi = gi_guard.as_slice_mut().expect("SumColumns backward: expected CPU buffer");

        debug_assert_eq!(go.len(), rows);
        debug_assert_eq!(gi.len(), rows * cols);

        for r in 0..rows {
            let g = go[r];
            for c in 0..cols {
                gi[c * rows + r] = g;
            }
        }
    }
}
