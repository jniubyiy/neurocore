use neurocore::tensor::Tensor3D;
use neurocore::jacobian::Jacobian3D;
use neurocore::layers::{Layer3D, Linear3D, Sigmoid3D};
use neurocore::loss_plan::{LossBlueprint, LossPlan};
use neurocore::dispatchers::single::loss::dim3d::SingleLoss3D;
use neurocore::dispatchers::common::model_trait::{Model3D, LossDispatch};
use neurocore::model_plan::param_store::{ParamStore, ParamSlice};

fn main() {
    let in_features = 4; let hidden = 2; let out_features = 4;
    let p_enc = in_features * hidden + hidden;       // 10
    let p_dec = hidden * out_features + out_features; // 12

    let mut store = ParamStore::new();
    let enc_slice = store.allocate_with(&vec![0.3; p_enc]);
    let dec_slice = store.allocate_with(&vec![0.3; p_dec]);

    let encoder = Linear3D::new(in_features, hidden, enc_slice);
    let sigmoid = Sigmoid3D::new();
    let decoder = Linear3D::new(hidden, out_features, dec_slice);

    let input = Tensor3D::new(vec![
        vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
    ]);
    let target = input.clone();
    let total_params = store.len();
    let j0 = Jacobian3D::new(input.depth, input.rows, in_features, total_params);

    let loss_plan = LossPlan::new(LossBlueprint::mse()).unwrap();
    let built_loss = loss_plan.build(out_features, out_features).unwrap();
    let loss_dispatch = SingleLoss3D::new();
    let lr = 0.1;

    let layers: Vec<std::sync::Arc<dyn Layer3D + Send + Sync>> = vec![
        std::sync::Arc::new(encoder),
        std::sync::Arc::new(sigmoid),
        std::sync::Arc::new(decoder),
    ];
    let slices = vec![enc_slice, ParamSlice::new(0, 0), dec_slice];
    let mut model = neurocore::dispatchers::auto::AutoModel3D::new(layers, slices, store, 1);

    for epoch in 0..500 {
        let (dec_out, j_dec) = model.forward(&input, &j0);
        let (loss, grad) = loss_dispatch.compute_loss(&dec_out, &target, &j_dec, &built_loss);
        model.update_params(lr, &grad);
        if epoch % 100 == 0 { println!("Epoch {}: loss={:.6}", epoch, loss); }
    }
    println!("\nОбучение завершено.");
}




