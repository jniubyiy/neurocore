use crate::tensor::Tensor1D;
use crate::jacobian::Jacobian;
use crate::neuron::base::Neuron;

pub struct Softmax;

impl Neuron for Softmax {
    fn forward(&self, input: &Tensor1D, j_input: &Jacobian) -> (Tensor1D, Jacobian) {
        let n = input.len();
        let p = j_input.cols();

        // Для численной стабильности вычитаем максимум
        let max_val = input.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps = vec![0.0; n];
        let mut sum_exp = 0.0;
        for i in 0..n {
            let exp_val = (input.data[i] - max_val).exp();
            exps[i] = exp_val;
            sum_exp += exp_val;
        }

        let mut out_data = vec![0.0; n];
        let mut j_out = Jacobian::new(n, p);

        for i in 0..n {
            let softmax_i = exps[i] / sum_exp;
            out_data[i] = softmax_i;
            // Вычисляем производные: для каждого параметра j
            // d(softmax_i)/d(input_k) = softmax_i * (delta_ik - softmax_k)
            // а d(softmax_i)/d(param_j) = sum_k [ d(softmax_i)/d(input_k) * d(input_k)/d(param_j) ]
            for j in 0..p {
                let mut deriv = 0.0;
                for k in 0..n {
                    let delta_ik = if i == k { 1.0 } else { 0.0 };
                    let dsoft_i_dinput_k = softmax_i * (delta_ik - exps[k] / sum_exp);
                    deriv += dsoft_i_dinput_k * j_input.data[k][j];
                }
                j_out.data[i][j] = deriv;
            }
        }
        (Tensor1D::new(out_data), j_out)
    }
}





