use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::UniversalLayer;
use crate::model_plan::param_store::ParamSlice;
use crate::linalg;
use faer::Mat;

pub struct SoftKeepGate {
    pub in_features: usize,
    pub temperature: f32,
}

impl SoftKeepGate {
    pub fn new(in_features: usize, temperature: f32) -> Self {
        assert!(temperature > 0.0, "SoftKeepGate: temperature must be positive");
        Self { in_features, temperature }
    }
}

impl UniversalLayer for SoftKeepGate {
    fn forward_mat(
        &self,
        input: &Mat<f32>,
        params: &[f32],
        slice: &ParamSlice,
    ) -> (Mat<f32>, DynamicContext) {
        let thresholds = &params[slice.start..slice.start + self.in_features];
        let tmp = self.temperature;

        let output = Mat::from_fn(input.nrows(), input.ncols(), |r, c| {
            let x = input[(r, c)];
            let z = (thresholds[c] - x.abs()) / tmp;
            x / (1.0 + (-z).exp())
        });

        let input_tensor = linalg::faer_to_tensor2d(input);
        let ctx = DynamicContext::Ctx1D(
            crate::layers::context1d::LayerContext1D::SoftKeepGate { input: input_tensor },
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
        let thresholds = &params[slice.start..slice.start + self.in_features];
        let tmp = self.temperature;

        let x_tensor = match ctx {
            DynamicContext::Ctx1D(c) => match c {
                crate::layers::context1d::LayerContext1D::SoftKeepGate { input } => input,
                _ => panic!("Expected SoftKeepGate context"),
            },
            _ => panic!("Expected Ctx1D context"),
        };
        let x_mat = linalg::tensor2d_to_faer(x_tensor);

        let (dx, d_thr) = soft_keep_gate_backward_mat(&x_mat, delta, thresholds, tmp);
        (dx, d_thr)
    }

    fn param_len(&self) -> usize { self.in_features }
    fn input_features(&self) -> usize { self.in_features }
    fn output_features(&self) -> usize { self.in_features }

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
        let thresholds = &params[slice.start..slice.start + self.in_features];
        let tmp = self.temperature;
        let chunk = input.submatrix(task_offset, 0, task_count, self.in_features);
        let out_chunk = Mat::from_fn(task_count, self.in_features, |r, c| {
            let x = chunk[(r, c)];
            let z = (thresholds[c] - x.abs()) / tmp;
            x / (1.0 + (-z).exp())
        });
        for r in 0..task_count {
            for c in 0..self.in_features {
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
            crate::layers::context1d::LayerContext1D::SoftKeepGate { input: t },
        )
    }

    fn output_mat_shape(&self, batch_size: usize) -> Mat<f32> {
        Mat::zeros(batch_size, self.in_features)
    }
}

fn soft_keep_gate_backward_mat(
    x: &Mat<f32>,
    dout: &Mat<f32>,
    thresholds: &[f32],
    temperature: f32,
) -> (Mat<f32>, Vec<f32>) {
    let rows = x.nrows();
    let cols = x.ncols();
    assert_eq!(cols, thresholds.len());

    let mut dx = Mat::zeros(rows, cols);
    let mut d_thr = vec![0.0f32; cols];

    for r in 0..rows {
        for c in 0..cols {
            let x_val = x[(r, c)];
            let abs_x = x_val.abs();
            let z = (thresholds[c] - abs_x) / temperature;
            let s = 1.0 / (1.0 + (-z).exp());
            let ds = s * (1.0 - s) / temperature;
            let df_dx = s - abs_x * ds;
            dx[(r, c)] = dout[(r, c)] * df_dx;

            let d_s_dthr = s * (1.0 - s) / temperature;
            d_thr[c] += dout[(r, c)] * x_val * d_s_dthr;
        }
    }

    (dx, d_thr)
}