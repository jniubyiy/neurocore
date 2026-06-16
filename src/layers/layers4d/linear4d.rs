use crate::layers::Linear3D;
use crate::layers::Layer3D;
use crate::tensor::Tensor4D;
use crate::jacobian::Jacobian4D;
use crate::model_plan::param_store::ParamSlice;
use crate::model_plan::blueprint::assert_power_of_two;
use super::Layer4D;

pub struct Linear4D {
    pub inner_3d: Linear3D,
}

impl Linear4D {
    pub fn new(in_features: usize, out_features: usize, slice: ParamSlice) -> Self {
        assert_power_of_two(in_features);
        assert_power_of_two(out_features);
        let inner_3d = Linear3D::new(in_features, out_features, slice);
        Self { inner_3d }
    }
}

impl Layer4D for Linear4D {
    fn forward_4d(
        &self,
        input: &Tensor4D,
        j_input: &Jacobian4D,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Tensor4D, Jacobian4D) {
        let dim1 = input.dim1;
        let depth = input.depth;
        let rows = input.rows;
        let total_params = j_input.num_params;
        let out_features = self.inner_3d.inner_2d.output_dim;
        let mut out_data = Vec::with_capacity(dim1);
        let mut j_out = Jacobian4D::new(dim1, depth, rows, out_features, total_params);

        for d in 0..dim1 {
            let slice_3d = input.slice_3d(d);
            let j_slice_3d = j_input.slice_jacobian(d);
            let (out_slice, j_out_slice) = self.inner_3d.forward_3d(&slice_3d, &j_slice_3d, params, slice);
            out_data.push(out_slice.data);
            j_out.set_slice_jacobian(d, &j_out_slice);
        }
        (Tensor4D::new(out_data), j_out)
    }

    fn param_len(&self) -> usize { self.inner_3d.param_len() }
}





