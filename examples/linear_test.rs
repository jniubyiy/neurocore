use neurocore::layers::{Layer, LinearLayer};
use neurocore::tensor::Tensor1D;
use neurocore::jacobian::Jacobian;
use neurocore::model_plan::param_store::{ParamStore, ParamSlice};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::dispatchers::single::loss::dim1d::SingleLoss1D;
use neurocore::dispatchers::single::SingleModel1D;
use neurocore::dispatchers::common::model_trait::{Model1D, LossDispatch};
use std::sync::Arc;

fn main() {
    let mut store = ParamStore::new();
    let slice = store.allocate_with(&vec![0.5; 10]); // 4*2 + 2
    let layer = LinearLayer::new(4, 2, slice);
    let x = Tensor1D::new(vec![1.0, 2.0, 3.0, 4.0]);
    let target = Tensor1D::new(vec![0.8, 1.5]);
    // Исправлено: out_features = 4 (длина входного вектора)
    let j_input = Jacobian::new(4, store.len());

    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build(2, 2).unwrap();
    let loss_dispatch = SingleLoss1D::new();
    let lr = 0.1;

    let layers: Vec<Arc<dyn Layer + Send + Sync>> = vec![Arc::new(layer)];
    let slices = vec![slice];
    let mut model = SingleModel1D::new(layers, slices, store);

    for epoch in 0..500 {
        let (pred, j_pred) = model.forward(&x, &j_input);
        let (loss, grad) = loss_dispatch.compute_loss(&pred, &target, &j_pred, &built_loss);
        model.update_params(lr, &grad);
        if epoch % 100 == 0 { println!("Epoch {}: loss={:.6}", epoch, loss); }
    }
    let (pred, _) = model.forward(&x, &j_input);
    println!("Final prediction: {:?}", pred.data);
}




