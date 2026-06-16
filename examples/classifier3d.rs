use neurocore::tensor::Tensor3D;
use neurocore::jacobian::Jacobian3D;
use neurocore::layers::{Layer3D, Linear3D, Softmax3D};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::dispatchers::single::loss::dim3d::SingleLoss3D;
use neurocore::dispatchers::common::model_trait::{Model3D, LossDispatch};
use neurocore::model_plan::param_store::{ParamStore, ParamSlice};

fn main() {
    let in_features = 2; let classes = 2;
    let total_params = in_features * classes + classes;
    let mut store = ParamStore::new();
    let slice = store.allocate_with(&vec![0.2; total_params]);
    let linear = Linear3D::new(in_features, classes, slice);
    let softmax = Softmax3D::new();

    let x = Tensor3D::new(vec![vec![vec![1.0, 2.0]]]);
    let y = Tensor3D::new(vec![vec![vec![0.0]]]);
    let j0 = Jacobian3D::new(1, 1, in_features, store.len());

    let loss_plan = LossPlan::new(LossBlueprint::cross_entropy(classes)).unwrap();
    let built_loss = loss_plan.build(classes, 1).unwrap();
    let loss_dispatch = SingleLoss3D::new();
    let lr = 0.5;

    let layers: Vec<std::sync::Arc<dyn Layer3D + Send + Sync>> = vec![
        std::sync::Arc::new(linear),
        std::sync::Arc::new(softmax),
    ];
    let slices = vec![slice, ParamSlice::new(0, 0)];
    let mut model = neurocore::dispatchers::auto::AutoModel3D::new(layers, slices, store, 1);

    for epoch in 0..200 {
        let (logits, jl) = model.forward(&x, &j0);
        let (loss, grad) = loss_dispatch.compute_loss(&logits, &y, &jl, &built_loss);
        model.update_params(lr, &grad);
        if epoch % 50 == 0 { println!("Epoch {}: loss={:.6}", epoch, loss); }
    }
    let (final_logits, _) = model.forward(&x, &j0);
    println!("\nFinal logits:\n{:?}", final_logits.data[0][0]);
}




