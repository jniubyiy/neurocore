use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::model_plan::param_store::ParamSlice;
use crate::model_plan::blueprint::assert_power_of_two;
use super::{Layer, LayerInfo};

pub struct LinearLayer {
    input_dim: usize,
    output_dim: usize,
    slice: ParamSlice,
}

impl LinearLayer {
    pub fn new(input_dim: usize, output_dim: usize, slice: ParamSlice) -> Self {
        assert_power_of_two(input_dim);
        assert_power_of_two(output_dim);
        assert_eq!(
            slice.len,
            input_dim * output_dim + output_dim,
            "LinearLayer: slice len must be in_dim*out_dim + out_dim"
        );
        Self { input_dim, output_dim, slice }
    }

    fn weight_index(&self, out_idx: usize, in_idx: usize) -> usize {
        self.slice.start + out_idx * self.input_dim + in_idx
    }

    fn bias_index(&self, out_idx: usize) -> usize {
        self.slice.start + self.input_dim * self.output_dim + out_idx
    }
}

impl Layer for LinearLayer {
    fn forward(
        &self,
        input: &Tensor1D,
        j_input: &Jacobian,
        params: &[f32],
        _slice: &ParamSlice,
    ) -> (Tensor1D, Jacobian) {
        let out_features = self.output_dim;
        let total_params = j_input.num_params;
        let mut out_data = vec![0.0; out_features];
        let mut j_out = Jacobian::new(out_features, total_params);

        for out_i in 0..out_features {
            let mut sum = 0.0;
            for in_i in 0..self.input_dim {
                let w = params[self.weight_index(out_i, in_i)];
                sum += w * input.data[in_i];

                let global_idx = self.weight_index(out_i, in_i);
                if global_idx < total_params {
                    j_out.data[out_i][global_idx] += input.data[in_i];
                }
            }
            let b = params[self.bias_index(out_i)];
            sum += b;
            let global_idx = self.bias_index(out_i);
            if global_idx < total_params {
                j_out.data[out_i][global_idx] += 1.0;
            }
            out_data[out_i] = sum;

            for p in 0..total_params {
                let mut deriv = 0.0;
                for in_i in 0..self.input_dim {
                    let w = params[self.weight_index(out_i, in_i)];
                    deriv += w * j_input.data[in_i][p];
                }
                j_out.data[out_i][p] += deriv;
            }
        }

        (Tensor1D::new(out_data), j_out)
    }

    fn param_len(&self) -> usize {
        self.slice.len
    }

    fn layer_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: "Linear".to_string(),
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            param_count: self.slice.len,
            param_start_index: Some(self.slice.start),
        }
    }
}





