use neurocore::layers::{Layer, LinearLayer, SoftmaxLayer};
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
    let layer_slice = store.allocate_with(&vec![0.2; 6]); // 2*2 веса + 2 bias

    let layer = LinearLayer::new(2, 2, layer_slice);
    let softmax = SoftmaxLayer::new();

    let x1 = Tensor1D::new(vec![1.0, 2.0]);
    let x2 = Tensor1D::new(vec![2.0, 1.0]);
    let y1 = Tensor1D::new(vec![0.0]);
    let y2 = Tensor1D::new(vec![1.0]);
    let total_params = store.len();
    let j0 = Jacobian::new(2, total_params);

    let loss_plan = LossPlan::new(LossBlueprint::cross_entropy(2)).unwrap();
    let built_loss = loss_plan.build(2, 1).unwrap();
    let loss_dispatch = SingleLoss1D::new();
    let lr = 0.5;

    let layers: Vec<Arc<dyn Layer + Send + Sync>> = vec![
        Arc::new(layer),
        Arc::new(softmax),
    ];
    let slices = vec![layer_slice, ParamSlice::new(0, 0)];
    let mut model = SingleModel1D::new(layers, slices, store);

    for e in 0..100 {
        for (x, y) in [(&x1, &y1), (&x2, &y2)] {
            let (logits, jl) = model.forward(x, &j0);
            let (loss, grad) = loss_dispatch.compute_loss(&logits, y, &jl, &built_loss);
            model.update_params(lr, &grad);
        }
        if e % 20 == 0 { println!("Epoch {}", e); }
    }
    println!("Done");
}





