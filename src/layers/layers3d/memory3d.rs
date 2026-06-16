use crate::layers::Memory2D;
use crate::layers::Layer2D;
use crate::tensor::Tensor3D;
use crate::jacobian::Jacobian3D;
use crate::model_plan::param_store::ParamSlice;
use crate::model_plan::blueprint::assert_power_of_two;
use super::Layer3D;

pub struct Memory3D {
    pub inner_2d: Memory2D,
}

impl Memory3D {
    pub fn new(in_features: usize, out_features: usize, slice: ParamSlice) -> Self {
        assert_power_of_two(in_features);
        assert_power_of_two(out_features);
        let inner_2d = Memory2D::new(in_features, out_features, slice);
        Self { inner_2d }
    }
}

impl Layer3D for Memory3D {
    fn forward_3d(
        &self,
        input: &Tensor3D,
        j_input: &Jacobian3D,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Tensor3D, Jacobian3D) {
        let depth = input.depth;
        let rows = input.rows;
        let total_params = j_input.num_params;
        let out_features = self.inner_2d.output_dim;
        let mut out_data = Vec::with_capacity(depth);
        let mut j_out = Jacobian3D::new(depth, rows, out_features, total_params);

        for d in 0..depth {
            let slice_2d = input.slice_2d(d);
            let j_slice_2d = j_input.slice_jacobian(d);
            let (out_slice, j_out_slice) = self.inner_2d.forward_2d(&slice_2d, &j_slice_2d, params, slice);
            out_data.push(out_slice.data);
            j_out.set_slice_jacobian(d, &j_out_slice);
        }
        (Tensor3D::new(out_data), j_out)
    }

    fn param_len(&self) -> usize { self.inner_2d.param_len() }
}