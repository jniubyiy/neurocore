// examples/expand_dim_example.rs
// Демонстрация Unsqueeze: Tensor2D (Dim1) -> Tensor3D (Dim2) -> обратно через ReduceMean.

use neurocore::compute_manager::DynamicTensor;
use neurocore::tensor::Tensor2D;
use neurocore::model_plan::{Plan, LayerDesc, LayerKind, Dim};

fn main() {
    // План: Unsqueeze(4 -> 2x2) затем ReduceMean(2x2 -> 4)
    let descs = vec![
        LayerDesc::new("unsqueeze", LayerKind::Unsqueeze(vec![2,2]), Dim::Dim1)
            .input(Dim::Dim1, &[4])
            .output(Dim::Dim2, &[2,2]),
        LayerDesc::new("reduce", LayerKind::ReduceMean(vec![2,2]), Dim::Dim2)
            .input(Dim::Dim2, &[2,2])
            .output(Dim::Dim1, &[4]),
    ];
    let plan = Plan::from_layer_descs(descs).expect("");
    let model = plan.build();

    let x = Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]]);
    let (y, _) = model.forward(DynamicTensor::Dim1(x.clone()));
    if let DynamicTensor::Dim1(t) = &y {
        println!("Input: {:?}", x.data);
        println!("Output after expand+reduce: {:?}", t.data);
    }
}