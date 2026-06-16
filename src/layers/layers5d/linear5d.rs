use crate::layers::Linear4D;
use crate::layers::Layer4D;
use crate::tensor::Tensor5D;
use crate::jacobian::Jacobian5D;
use crate::model_plan::param_store::ParamSlice;
use crate::model_plan::blueprint::assert_power_of_two;
use super::Layer5D;

pub struct Linear5D {
    pub inner_4d: Linear4D,
}

impl Linear5D {
    pub fn new(in_features: usize, out_features: usize, slice: ParamSlice) -> Self {
        assert_power_of_two(in_features);
        assert_power_of_two(out_features);
        let inner_4d = Linear4D::new(in_features, out_features, slice);
        Self { inner_4d }
    }
}

impl Layer5D for Linear5D {
    fn forward_5d(
        &self,
        input: &Tensor5D,
        j_input: &Jacobian5D,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Tensor5D, Jacobian5D) {
        let outer = input.outer;
        let dim1 = input.dim1;
        let depth = input.depth;
        let rows = input.rows;
        let total_params = j_input.num_params;
        let out_features = self.inner_4d.inner_3d.inner_2d.output_dim;
        let mut out_data = Vec::with_capacity(outer);
        let mut j_out = Jacobian5D::new(outer, dim1, depth, rows, out_features, total_params);

        for o in 0..outer {
            let slice_4d = input.slice_4d(o);
            let j_slice_4d = j_input.slice_jacobian(o);
            let (out_slice, j_out_slice) = self.inner_4d.forward_4d(&slice_4d, &j_slice_4d, params, slice);
            out_data.push(out_slice.data);
            j_out.set_slice_jacobian(o, &j_out_slice);
        }
        (Tensor5D::new(out_data), j_out)
    }

    fn param_len(&self) -> usize { self.inner_4d.param_len() }
}





