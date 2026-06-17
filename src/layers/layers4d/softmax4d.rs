use crate::tensor::Tensor4D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Softmax;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer4D, LayerContext4D};

pub struct Softmax4D { inner_size: usize }
impl Softmax4D { pub fn new(size: usize) -> Self { Self { inner_size: size } } }

impl Layer4D for Softmax4D {
    fn forward_into(&self, input: &Tensor4D, _params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<Vec<f32>>>>) -> LayerContext4D {
        let mat = linalg::tensor4d_to_faer(input);
        let out = Softmax.forward_mat(&mat); // softmax по строкам матрицы (все сгруппированные строки)
        let t = linalg::faer_to_tensor4d(&out, input.dim1, input.depth, input.rows, input.cols);
        *out_buf = t.data.clone();
        LayerContext4D::Softmax4D { output: t }
    }

    fn backward(&self, ctx: &LayerContext4D, delta: &Tensor4D, _params: &[f32], _slice: &ParamSlice) -> (Tensor4D, Vec<f32>) {
        let sm = match ctx { LayerContext4D::Softmax4D { output } => output, _ => panic!() };
        let (dim1, depth, rows, cols) = (sm.dim1, sm.depth, sm.rows, sm.cols);
        let mut d_prev = vec![vec![vec![vec![0.0; cols]; rows]; depth]; dim1];
        for d1 in 0..dim1 { for d in 0..depth { for r in 0..rows {
            for i in 0..cols {
                let mut sum = 0.0;
                for j in 0..cols {
                    let kron = if i == j { 1.0 } else { 0.0 };
                    sum += delta.data[d1][d][r][j] * sm.data[d1][d][r][j] * (kron - sm.data[d1][d][r][i]);
                }
                d_prev[d1][d][r][i] = sum;
            }
        }}}
        (Tensor4D::new(d_prev), vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn in_features(&self) -> usize { self.inner_size }
    fn out_features(&self) -> usize { self.inner_size }
}



