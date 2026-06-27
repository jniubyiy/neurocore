// examples/reduce_dim_example.rs
// Демонстрация ReduceMean: Tensor3D -> Tensor2D -> обратно через Unsqueeze.

use neurocore::compute_manager::DynamicTensor;
use neurocore::tensor::Tensor3D;
use neurocore::model_plan::{Plan, LayerDesc, LayerKind, Dim};

fn main() {
    let descs = vec![
        LayerDesc::new("reduce", LayerKind::ReduceMean(vec![2,2]), Dim::Dim2)
            .input(Dim::Dim2, &[2,2])
            .output(Dim::Dim1, &[4]),
        LayerDesc::new("unsqueeze", LayerKind::Unsqueeze(vec![2,2]), Dim::Dim1)
            .input(Dim::Dim1, &[4])
            .output(Dim::Dim2, &[2,2]),
    ];
    let plan = Plan::from_layer_descs(descs).expect("");
    let model = plan.build();

    let x = Tensor3D::new(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
    let (y, _) = model.forward(DynamicTensor::Dim2(x.clone()));
    if let DynamicTensor::Dim2(t) = &y {
        println!("Input: {:?}", x.data);
        println!("Output after reduce+expand: {:?}", t.data);
    }
}