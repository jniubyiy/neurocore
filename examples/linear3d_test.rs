use neurocore::tensor::Tensor3D;
use neurocore::jacobian::Jacobian3D;
use neurocore::layers::{Layer3D, Linear3D};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::dispatchers::single::loss::dim3d::SingleLoss3D;
use neurocore::dispatchers::common::model_trait::{Model3D, LossDispatch};
use neurocore::model_plan::param_store::{ParamStore, ParamSlice};

fn main() {
    let mut store = ParamStore::new();
    let slice = store.allocate_with(&vec![0.5; 10]); // 4*2 + 2
    let layer = Linear3D::new(4, 2, slice);
    let input = Tensor3D::new(vec![vec![vec![1.0, 2.0, 3.0, 4.0]]]);
    let target = Tensor3D::new(vec![vec![vec![0.8, 1.2]]]);
    let j = Jacobian3D::new(1, 1, 4, store.len());

    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build(2, 2).unwrap();
    let loss_dispatch = SingleLoss3D::new();
    let lr = 0.01;

    let layers: Vec<std::sync::Arc<dyn Layer3D + Send + Sync>> = vec![std::sync::Arc::new(layer)];
    let slices = vec![slice];
    let mut model = neurocore::dispatchers::auto::AutoModel3D::new(layers, slices, store, 1);

    for e in 0..500 {
        let (pred, jp) = model.forward(&input, &j);
        let (loss, grad) = loss_dispatch.compute_loss(&pred, &target, &jp, &built_loss);
        model.update_params(lr, &grad);
        if e % 100 == 0 { println!("Epoch {}: loss={:.6}", e, loss); }
    }
    let (pred, _) = model.forward(&input, &j);
    println!("Final prediction:\n{:?}", pred.data);
}