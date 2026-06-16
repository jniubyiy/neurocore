use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::model_plan::param_store::ParamSlice;
use crate::model_plan::blueprint::assert_power_of_two;
use super::{Layer, LayerInfo};

pub struct MemoryLayer {
    input_dim: usize,
    output_dim: usize,
    slice: ParamSlice,
}

impl MemoryLayer {
    pub fn new(input_dim: usize, output_dim: usize, slice: ParamSlice) -> Self {
        assert_power_of_two(input_dim);
        assert_power_of_two(output_dim);
        assert_eq!(
            slice.len,
            output_dim * (2 * input_dim + 1),
            "MemoryLayer: incorrect slice length"
        );
        Self { input_dim, output_dim, slice }
    }
}

impl Layer for MemoryLayer {
    fn forward(
        &self,
        input: &Tensor1D,
        j_input: &Jacobian,
        params: &[f32],
        _slice: &ParamSlice,
    ) -> (Tensor1D, Jacobian) {
        let total_params = j_input.num_params;
        let mut out_data = vec![0.0; self.output_dim];
        let mut j_out = Jacobian::new(self.output_dim, total_params);

        for out_i in 0..self.output_dim {
            let offset = self.slice.start + out_i * (2 * self.input_dim + 1);
            let m0: Vec<f32> = (0..self.input_dim)
                .map(|i| params[offset + i])
                .collect();
            let m1: Vec<f32> = (0..self.input_dim)
                .map(|i| params[offset + self.input_dim + i])
                .collect();
            let t_val = params[offset + 2 * self.input_dim];

            let mut dot0 = 0.0;
            let mut dot1 = 0.0;
            for i in 0..self.input_dim {
                dot0 += input.data[i] * m0[i];
                dot1 += input.data[i] * m1[i];
            }

            let logit0 = dot0 / t_val;
            let logit1 = dot1 / t_val;
            let max_logit = logit0.max(logit1);
            let exp0 = (logit0 - max_logit).exp();
            let exp1 = (logit1 - max_logit).exp();
            let sum_exp = exp0 + exp1;
            let soft0 = exp0 / sum_exp;
            let soft1 = exp1 / sum_exp;

            let y_val = soft0 * dot0 + soft1 * dot1;
            out_data[out_i] = y_val;

            let ds0_dot0 = soft0 * (1.0 - soft0) / t_val;
            let ds0_dot1 = -soft0 * soft1 / t_val;
            let ds1_dot0 = -soft1 * soft0 / t_val;
            let ds1_dot1 = soft1 * (1.0 - soft1) / t_val;

            let dy_dot0 = soft0 + dot0 * ds0_dot0 + dot1 * ds1_dot0;
            let dy_dot1 = soft1 + dot0 * ds0_dot1 + dot1 * ds1_dot1;

            for p in 0..total_params {
                let mut grad = 0.0;
                for k in 0..self.input_dim {
                    let dy_dxk = dy_dot0 * m0[k] + dy_dot1 * m1[k];
                    grad += dy_dxk * j_input.data[k][p];
                }
                for k in 0..self.input_dim {
                    let idx = offset + k;
                    if idx < total_params && p == idx {
                        grad += dy_dot0 * input.data[k];
                    }
                }
                for k in 0..self.input_dim {
                    let idx = offset + self.input_dim + k;
                    if idx < total_params && p == idx {
                        grad += dy_dot1 * input.data[k];
                    }
                }
                let avg_dot = soft0 * dot0 + soft1 * dot1;
                let ds0_dt = soft0 * (dot0 - avg_dot) / (t_val * t_val);
                let ds1_dt = soft1 * (dot1 - avg_dot) / (t_val * t_val);
                let dy_dt = dot0 * ds0_dt + dot1 * ds1_dt;
                let idx_t = offset + 2 * self.input_dim;
                if idx_t < total_params && p == idx_t {
                    grad += dy_dt;
                }
                j_out.data[out_i][p] += grad;
            }
        }

        (Tensor1D::new(out_data), j_out)
    }

    fn param_len(&self) -> usize { self.slice.len }

    fn layer_info(&self) -> LayerInfo {
        LayerInfo {
            layer_type: "Memory".to_string(),
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            param_count: self.slice.len,
            param_start_index: Some(self.slice.start),
        }
    }
}