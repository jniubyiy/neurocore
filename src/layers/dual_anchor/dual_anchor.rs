use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use faer::Mat;

pub struct DualAnchor {
    features: usize,
}

impl DualAnchor {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert_eq!(in_features, out_features,
            "DualAnchor: in_features must equal out_features");
        Self { features: in_features }
    }

    fn get_params(&self, params: &[f32], slice: &ParamSlice) -> (Vec<f32>, Vec<f32>, f32) {
        let base = slice.start;
        let min_vals = params[base..base + self.features].to_vec();
        let max_vals = params[base + self.features..base + 2 * self.features].to_vec();
        let alpha = params[base + 2 * self.features];
        (min_vals, max_vals, alpha)
    }
}

impl UniversalLayer for DualAnchor {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let (min_vals, max_vals, alpha) = self.get_params(params, slice);
        let batch = input.nrows();
        let features = self.features;

        let output = Mat::from_fn(batch, features, |r, c| {
            let x = input[(r, c)];
            let min_v = min_vals[c];
            let max_v = max_vals[c];
            let d_min = (x - min_v).abs();
            let d_max = (x - max_v).abs();
            let closest = if d_min <= d_max { min_v } else { max_v };
            x + alpha * (closest - x)
        });

        let input_tensor = linalg::faer_to_tensor2d(input);
        let ctx = DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::DualAnchor1D { input: input_tensor },
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
        let (min_vals, max_vals, alpha) = self.get_params(params, slice);
        let x_tensor = match ctx {
            DynamicContext::Ctx1D(c) => match c {
                crate::layers::context1d::LayerContext1D::DualAnchor1D { input } => input,
                _ => panic!("Expected DualAnchor1D context"),
            },
            _ => panic!("Expected Ctx1D context"),
        };
        let x_mat = linalg::tensor2d_to_faer(x_tensor);

        let (dx, grad) = dual_anchor_backward_mat(&x_mat, delta, &min_vals, &max_vals, alpha);
        (dx, grad)
    }

    fn param_len(&self) -> usize {
        2 * self.features + 1
    }
    fn input_features(&self) -> usize { self.features }
    fn output_features(&self) -> usize { self.features }

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
        let (min_vals, max_vals, alpha) = self.get_params(params, slice);
        let chunk = input.submatrix(task_offset, 0, task_count, self.features);
        let out_chunk = Mat::from_fn(task_count, self.features, |r, c| {
            let x = chunk[(r, c)];
            let min_v = min_vals[c];
            let max_v = max_vals[c];
            let d_min = (x - min_v).abs();
            let d_max = (x - max_v).abs();
            let closest = if d_min <= d_max { min_v } else { max_v };
            x + alpha * (closest - x)
        });
        for r in 0..task_count {
            for c in 0..self.features {
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
            crate::layers::context1d::LayerContext1D::DualAnchor1D { input: t },
        )
    }

    fn output_mat_shape(&self, batch_size: usize) -> Mat<f32> {
        Mat::zeros(batch_size, self.features)
    }
}

fn dual_anchor_backward_mat(
    x: &Mat<f32>,
    dout: &Mat<f32>,
    min_vals: &[f32],
    max_vals: &[f32],
    alpha: f32,
) -> (Mat<f32>, Vec<f32>) {
    let batch = x.nrows();
    let features = x.ncols();

    let mut dx = Mat::zeros(batch, features);
    let mut d_min = vec![0.0f32; features];
    let mut d_max = vec![0.0f32; features];
    let mut d_alpha = 0.0f32;

    for r in 0..batch {
        for c in 0..features {
            let x_val = x[(r, c)];
            let min_v = min_vals[c];
            let max_v = max_vals[c];
            let d_min_abs = (x_val - min_v).abs();
            let d_max_abs = (x_val - max_v).abs();
            let is_min = d_min_abs <= d_max_abs;
            let gout = dout[(r, c)];

            dx[(r, c)] += gout * (1.0 - alpha);

            if is_min {
                d_min[c] += gout * alpha;
                d_alpha += gout * (min_v - x_val);
            } else {
                d_max[c] += gout * alpha;
                d_alpha += gout * (max_v - x_val);
            }
        }
    }

    let mut grad = Vec::with_capacity(2 * features + 1);
    grad.extend_from_slice(&d_min);
    grad.extend_from_slice(&d_max);
    grad.push(d_alpha);
    (dx, grad)
}