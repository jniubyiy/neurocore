use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use faer::Mat;

pub struct Linear {
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self { in_features, out_features }
    }

    pub(crate) fn get_weight_matrix_and_bias(
        &self,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>) {
        let in_feat = self.in_features;
        let out_feat = self.out_features;
        let w_start = slice.start;
        let b_start = w_start + in_feat * out_feat;

        let weight = Mat::from_fn(out_feat, in_feat, |r, c| {
            params[w_start + r * in_feat + c]
        });
        let bias = params[b_start..b_start + out_feat].to_vec();
        (weight, bias)
    }
}

impl UniversalLayer for Linear {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let (weight, bias) = self.get_weight_matrix_and_bias(params, slice);
        let batch = input.nrows();
        let mut output = input * weight.transpose();
        output += Mat::from_fn(batch, self.out_features, |_, j| bias[j]);

        let input_tensor = linalg::faer_to_tensor2d(input);
        let ctx = DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::Linear { input: input_tensor },
        );
        (output, ctx)
    }

    fn backward_mat(
        &self,
        ctx: &DynamicContext,
        delta: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, Vec<f32>) {
        let x_tensor = match ctx {
            DynamicContext::Ctx1D(c) => match c {
                crate::layers::context1d::LayerContext1D::Linear { input } => input,
                _ => panic!("Expected Linear context"),
            },
            _ => panic!("Expected Ctx1D context"),
        };
        let x_mat = linalg::tensor2d_to_faer(x_tensor);
        let (weight, _) = self.get_weight_matrix_and_bias(params, slice);
        let dx = delta * &weight;
        let dw = delta.transpose() * &x_mat;
        let batch = delta.nrows();
        let mut db = vec![0.0f32; self.out_features];
        for r in 0..batch {
            for c in 0..self.out_features {
                db[c] += delta[(r, c)];
            }
        }

        let mut grad = Vec::with_capacity(self.param_len());
        for r in 0..self.out_features {
            for c in 0..self.in_features {
                grad.push(dw[(r, c)]);
            }
        }
        grad.extend_from_slice(&db);
        (dx, grad)
    }

    fn param_len(&self) -> usize {
        self.in_features * self.out_features + self.out_features
    }

    fn input_features(&self) -> usize { self.in_features }
    fn output_features(&self) -> usize { self.out_features }

    fn total_tasks(&self, batch_size: usize) -> usize { batch_size }

    fn execute_tasks(
        &self,
        input: &Mat<f32>,
        output: &mut Mat<f32>,
        task_offset: usize,
        task_count: usize,
        params: &[f32],
        slice: &ParamSlice,
    ) {
        let (weight, bias) = self.get_weight_matrix_and_bias(params, slice);
        let chunk = input.submatrix(task_offset, 0, task_count, self.in_features);
        let mut out_chunk = &chunk * weight.transpose();
        out_chunk += Mat::from_fn(task_count, self.out_features, |_, j| bias[j]);
        for r in 0..task_count {
            for c in 0..self.out_features {
                output[(task_offset + r, c)] = out_chunk[(r, c)];
            }
        }
    }

    fn create_sample_context(
        &self,
        input_sample: &Mat<f32>,
        _output_sample: &Mat<f32>,
    ) -> DynamicContext {
        let t = linalg::faer_to_tensor2d(input_sample);
        DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::Linear { input: t },
        )
    }

    fn output_mat_shape(&self, batch_size: usize) -> Mat<f32> {
        Mat::zeros(batch_size, self.out_features)
    }

    fn as_linear(&self) -> Option<&Linear> {
        Some(self)
    }
}