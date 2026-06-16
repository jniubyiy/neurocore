use neurocore::tensor::Tensor5D;
use neurocore::jacobian::Jacobian5D;
use neurocore::layers::{Layer5D, Linear5D};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::dispatchers::single::loss::dim5d::SingleLoss5D;
use neurocore::dispatchers::common::model_trait::{Model5D, LossDispatch};
use neurocore::model_plan::param_store::{ParamStore, ParamSlice};

fn main() {
    let mut store = ParamStore::new();
    let slice = store.allocate_with(&vec![0.5; 10]); // 4*2 + 2
    let layer = Linear5D::new(4, 2, slice);
    let input = Tensor5D::new(vec![vec![vec![vec![vec![1.0, 2.0, 3.0, 4.0]]]]]);
    let target = Tensor5D::new(vec![vec![vec![vec![vec![0.8, 1.2]]]]]);
    let j = Jacobian5D::new(1, 1, 1, 1, 4, store.len());

    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build(2, 2).unwrap();
    let loss_dispatch = SingleLoss5D::new();
    let lr = 0.01;

    let layers: Vec<std::sync::Arc<dyn Layer5D + Send + Sync>> = vec![std::sync::Arc::new(layer)];
    let slices = vec![slice];
    let mut model = neurocore::dispatchers::auto::AutoModel5D::new(layers, slices, store, 1);

    for e in 0..500 {
        let (pred, jp) = model.forward(&input, &j);
        let (loss, grad) = loss_dispatch.compute_loss(&pred, &target, &jp, &built_loss);
        model.update_params(lr, &grad);
        if e % 100 == 0 { println!("Epoch {}: loss={:.6}", e, loss); }
    }
}




