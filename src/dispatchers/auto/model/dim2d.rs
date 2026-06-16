use crate::layers::Layer2D;
use crate::tensor::Tensor2D;
use crate::jacobian::Jacobian2D;
use crate::model_plan::param_store::{ParamSlice, ParamStore};
use crate::dispatchers::common::{Model2D,
    flatten::{flatten_tensor, flatten_jacobian, unflatten_tensor, unflatten_jacobian}
};
use std::sync::Arc;
use std::thread;

pub struct AutoModel2D {
    layers: Vec<Arc<dyn Layer2D + Send + Sync>>,
    slices: Vec<ParamSlice>,
    store: ParamStore,
    num_workers: usize,
}

impl AutoModel2D {
    pub fn new(
        layers: Vec<Arc<dyn Layer2D + Send + Sync>>,
        slices: Vec<ParamSlice>,
        store: ParamStore,
        num_threads: usize,
    ) -> Self {
        assert_eq!(layers.len(), slices.len());
        Self { layers, slices, store, num_workers: num_threads.max(1) }
    }
}

impl Model2D for AutoModel2D {
    fn forward(&mut self, input: &Tensor2D, j_input: &Jacobian2D) -> (Tensor2D, Jacobian2D) {
        let batch_rows = input.rows;
        let total_params = j_input.num_params;          // <-- исправлено
        let params = self.store.all_params();

        let mut val_flat = flatten_tensor(input);
        let mut jac_flat = flatten_jacobian(j_input);

        for (layer_idx, (layer, slice)) in self.layers.iter().zip(self.slices.iter()).enumerate() {
            let in_rows = batch_rows;
            let in_cols = if layer_idx == 0 { input.cols } else { self.layers[layer_idx-1].out_features() };
            let out_cols = layer.out_features();

            let mut next_val = vec![0.0f32; in_rows * out_cols];
            let mut next_jac = vec![0.0f32; in_rows * out_cols * total_params];

            let num_threads = self.num_workers.min(in_rows);
            let rows_per = (in_rows + num_threads - 1) / num_threads;

            let val_flat_arc = Arc::new(val_flat.clone());
            let jac_flat_arc = Arc::new(jac_flat.clone());
            let params_arc = Arc::new(params.to_vec());
            let layer_arc = Arc::clone(layer);
            let slice = *slice;

            let next_val_ptr = next_val.as_ptr() as usize;
            let next_jac_ptr = next_jac.as_ptr() as usize;

            let mut handles = vec![];
            for t in 0..num_threads {
                let row_start = t * rows_per;
                let row_end = (row_start + rows_per).min(in_rows);
                if row_start >= row_end { continue; }
                let val_flat = Arc::clone(&val_flat_arc);
                let jac_flat = Arc::clone(&jac_flat_arc);
                let params = Arc::clone(&params_arc);
                let layer = Arc::clone(&layer_arc);
                let slice = slice;

                let handle = thread::spawn(move || {
                    let in_tensor = unflatten_tensor(&val_flat, in_rows, in_cols);
                    let j_in_tensor = unflatten_jacobian(&jac_flat, in_rows, in_cols, total_params);
                    let out_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            (next_val_ptr as *mut f32).add(row_start * out_cols),
                            (row_end - row_start) * out_cols,
                        )
                    };
                    let jac_out_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            (next_jac_ptr as *mut f32).add(row_start * out_cols * total_params),
                            (row_end - row_start) * out_cols * total_params,
                        )
                    };
                    layer.execute_range(
                        &in_tensor, &j_in_tensor,
                        out_slice, jac_out_slice,
                        row_start, row_end, 0, out_cols,
                        total_params, &params, &slice,
                    );
                });
                handles.push(handle);
            }
            for h in handles { h.join().unwrap(); }

            val_flat = next_val;
            jac_flat = next_jac;
        }

        let last_out_cols = self.layers.last().unwrap().out_features();
        let out_tensor = unflatten_tensor(&val_flat, batch_rows, last_out_cols);
        let out_jacob = unflatten_jacobian(&jac_flat, batch_rows, last_out_cols, total_params);
        (out_tensor, out_jacob)
    }

    fn update_params(&mut self, lr: f32, grad: &[f32]) {
        self.store.apply_gradient(lr, grad);
    }

    fn num_workers(&self) -> usize { self.num_workers }
}