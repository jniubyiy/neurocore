use crate::model_plan::param_store::ParamSlice;
use faer::Mat;

pub struct Combiner {
    input_dim: usize,
    output_dim: usize,
}

impl Combiner {
    pub fn new(input_dims: Vec<usize>, output_dim: usize) -> Self {
        assert_eq!(input_dims.len(), 2, "Combiner requires exactly two inputs");
        assert!(input_dims[0] == input_dims[1], "Combiner inputs must have same size for now");
        Self { input_dim: input_dims[0], output_dim }
    }

    pub fn input_dim(&self) -> usize { self.input_dim }
    pub fn output_dim(&self) -> usize { self.output_dim }

    /// Публичный метод для получения матриц весов и смещения из слайса параметров.
    pub fn get_weights_and_bias(&self, params: &[f32], slice: &ParamSlice) -> (Mat<f32>, Mat<f32>, Vec<f32>) {
        let in_feat = self.input_dim;
        let out_feat = self.output_dim;
        let base = slice.start;

        let wa = Mat::from_fn(out_feat, in_feat, |r, c| {
            params[base + r * in_feat + c]
        });
        let wb_offset = base + out_feat * in_feat;
        let wb = Mat::from_fn(out_feat, in_feat, |r, c| {
            params[wb_offset + r * in_feat + c]
        });
        let bias_offset = wb_offset + out_feat * in_feat;
        let bias = params[bias_offset..bias_offset + out_feat].to_vec();
        (wa, wb, bias)
    }

    /// Возвращает параметры combiner в виде плоских Vec<f32>.
    /// Порядок:
    /// - `wa`: row‑major, длина `out_feat * in_feat`
    /// - `wb`: row‑major, длина `out_feat * in_feat`
    /// - `bias`: длина `out_feat`
    pub(crate) fn get_weights_and_bias_vec(
        &self,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let in_feat = self.input_dim;
        let out_feat = self.output_dim;
        let base = slice.start;

        let wa_len = out_feat * in_feat;
        let wa = params[base..base + wa_len].to_vec();

        let wb_start = base + wa_len;
        let wb = params[wb_start..wb_start + wa_len].to_vec();

        let bias_start = wb_start + wa_len;
        let bias = params[bias_start..bias_start + out_feat].to_vec();

        (wa, wb, bias)
    }

    /// Прямой матричный проход: output = ReLU( a @ Wa^T + b @ Wb^T + bias )
    pub fn forward_mat(
        &self,
        a: &Mat<f32>,
        b: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> Mat<f32> {
        let (wa, wb, bias) = self.get_weights_and_bias(params, slice);
        let batch = a.nrows();
        let out_feat = self.output_dim;

        let mut pre = a * wa.transpose() + b * wb.transpose();
        pre += Mat::from_fn(batch, out_feat, |_, j| bias[j]);
        pre.map(|x| x.max(0.0))
    }

    /// Обратный матричный проход.
    pub fn backward_mat(
        &self,
        a: &Mat<f32>,
        b: &Mat<f32>,
        d_out: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, Mat<f32>, Vec<f32>) {
        let (wa, wb, bias) = self.get_weights_and_bias(params, slice);
        let rows = a.nrows();
        let m = self.output_dim;

        let mut pre = a * wa.transpose() + b * wb.transpose();
        for r in 0..rows {
            for c in 0..m {
                pre[(r, c)] += bias[c];
            }
        }

        let d_pre = Mat::from_fn(rows, m, |r, c| {
            if pre[(r, c)] > 0.0 { d_out[(r, c)] } else { 0.0 }
        });

        let da = &d_pre * &wa;
        let db = &d_pre * &wb;

        let d_wa = &d_pre.transpose() * a;
        let d_wb = &d_pre.transpose() * b;
        let mut d_bias = vec![0.0f32; m];
        for r in 0..rows {
            for c in 0..m {
                d_bias[c] += d_pre[(r, c)];
            }
        }

        let mut grad = Vec::with_capacity(self.param_len());
        for r in 0..m {
            for c in 0..self.input_dim {
                grad.push(d_wa[(r, c)]);
            }
        }
        for r in 0..m {
            for c in 0..self.input_dim {
                grad.push(d_wb[(r, c)]);
            }
        }
        grad.extend_from_slice(&d_bias);

        (da, db, grad)
    }

    pub fn param_len(&self) -> usize {
        2 * self.output_dim * self.input_dim + self.output_dim
    }
}