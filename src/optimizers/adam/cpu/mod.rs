use std::any::Any;
use std::sync::atomic::Ordering;

use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::plans::optimizer_plan::cube::OptimizerCube;

use super::super::adam::Adam;

impl OptimizerCube for Adam {
    fn state_size_per_param(&self) -> usize {
        2
    }

    fn apply_buffered_handle(
        &self,
        _params: &MatrixBufferHandle,
        grads: &MatrixBufferHandle,
        state: &MatrixBufferHandle,
    ) {
        assert!(
            !grads.is_gpu() && !state.is_gpu(),
            "Adam: grads and state must be CPU"
        );
        let n = grads.rows() * grads.cols();

        let mut grad_guard = grads.write();
        let g_slice = grad_guard.as_slice_mut().expect("Adam: expected CPU buffer");
        let mut state_guard = state.write();
        let s_slice = state_guard.as_slice_mut().expect("Adam: expected CPU buffer");

        debug_assert_eq!(s_slice.len(), n * 2);

        let (m_slice, v_slice) = s_slice.split_at_mut(n);

        let t = self.step_counter.fetch_add(1, Ordering::SeqCst) + 1;

        let bias_correction1 = 1.0 - self.beta1.powi(t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(t as i32);

        for i in 0..n {
            m_slice[i] = self.beta1 * m_slice[i] + (1.0 - self.beta1) * g_slice[i];
            v_slice[i] = self.beta2 * v_slice[i] + (1.0 - self.beta2) * g_slice[i] * g_slice[i];

            let m_hat = m_slice[i] / bias_correction1;
            let v_hat = v_slice[i] / bias_correction2;

            g_slice[i] = m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}