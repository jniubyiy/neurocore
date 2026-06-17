use crate::tensor::Tensor3D;
use crate::model_plan::param_store::ParamSlice;
use crate::neuron::Softmax;
use crate::neuron::base::Neuron;
use crate::linalg;
use super::{Layer3D, LayerContext3D};

pub struct Softmax3D {
    inner_size: usize,
}

impl Softmax3D {
    pub fn new(size: usize) -> Self { Self { inner_size: size } }
}

impl Layer3D for Softmax3D {
    fn forward_into(&self, input: &Tensor3D, _params: &[f32], _slice: &ParamSlice, out_buf: &mut Vec<Vec<Vec<f32>>>) -> LayerContext3D {
        // Softmax должен применяться к каждой строке отдельно.
        // Но в 3D тензоре «строкой» считается последняя размерность (cols).
        // Мы уже преобразовали тензор в матрицу (total_rows × cols), и Softmax.forward_mat
        // применяет softmax к каждой строке этой матрицы. Это как раз то, что нужно.
        let mat = linalg::tensor3d_to_faer(input);
        let out = Softmax.forward_mat(&mat);
        let t = linalg::faer_to_tensor3d(&out, input.depth, input.rows, input.cols);
        *out_buf = t.data.clone();
        LayerContext3D::Softmax3D { output: t }
    }

    fn backward(&self, ctx: &LayerContext3D, delta: &Tensor3D, _params: &[f32], _slice: &ParamSlice) -> (Tensor3D, Vec<f32>) {
        let sm = match ctx { LayerContext3D::Softmax3D { output } => output, _ => panic!() };
        let depth = sm.depth;
        let rows = sm.rows;
        let cols = sm.cols;
        let mut d_prev = vec![vec![vec![0.0; cols]; rows]; depth];
        for d in 0..depth {
            for r in 0..rows {
                for i in 0..cols {
                    let mut sum = 0.0;
                    for j in 0..cols {
                        let kron = if i == j { 1.0 } else { 0.0 };
                        sum += delta.data[d][r][j] * sm.data[d][r][j] * (kron - sm.data[d][r][i]);
                    }
                    d_prev[d][r][i] = sum;
                }
            }
        }
        (Tensor3D::new(d_prev), vec![])
    }

    fn param_len(&self) -> usize { 0 }
    fn in_features(&self) -> usize { self.inner_size }
    fn out_features(&self) -> usize { self.inner_size }
}





