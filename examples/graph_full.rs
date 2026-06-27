// examples/graph_full.rs
// Полный граф в стиле graph_model.rs: Splitter, SplitterConnector, CombinerConnector, Combiner.

use neurocore::model_plan::Plan;
use neurocore::compute_manager::DynamicTensor;
use neurocore::tensor::Tensor2D;
use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub};
use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};

mod models {
    use neurocore::model_plan::{LayerDesc, LayerKind, Dim};

    // Входной Linear + Splitter + SplitterConnector
    pub fn stage1() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("input", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[6])
                .output(Dim::Dim1, &[6]),
            LayerDesc::new("splitter", LayerKind::Splitter, Dim::Dim1)
                .input(Dim::Dim1, &[6])
                .output(Dim::Dim1, (&[2], &[4])),
            LayerDesc::new("split_conn", LayerKind::SplitterConnector, Dim::Dim1)
                .input(Dim::Dim1, (&[2], &[4])),
        ]
    }

    // Ветка A (2 → 3)
    pub fn branch_a() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("a_in", LayerKind::SplitterConnector, Dim::Dim1)
                .output(Dim::Dim1, &[2]),
            LayerDesc::new("a_linear", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[2])
                .output(Dim::Dim1, &[3]),
            LayerDesc::new("a_relu", LayerKind::ReLU, Dim::Dim1)
                .input(Dim::Dim1, &[3])
                .output(Dim::Dim1, &[3]),
            LayerDesc::new("a_out", LayerKind::CombinerConnector, Dim::Dim1)
                .input(Dim::Dim1, &[3]),
        ]
    }

    // Ветка B (4 → 3)
    pub fn branch_b() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("b_in", LayerKind::SplitterConnector, Dim::Dim1)
                .output(Dim::Dim1, &[4]),
            LayerDesc::new("b_linear", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[4])
                .output(Dim::Dim1, &[3]),
            LayerDesc::new("b_sigmoid", LayerKind::Sigmoid, Dim::Dim1)
                .input(Dim::Dim1, &[3])
                .output(Dim::Dim1, &[3]),
            LayerDesc::new("b_out", LayerKind::CombinerConnector, Dim::Dim1)
                .input(Dim::Dim1, &[3]),
        ]
    }

    // Объединяющая часть: CombinerConnector + Combiner + выход
    pub fn stage2() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("comb_conn_in", LayerKind::CombinerConnector, Dim::Dim1)
                .output(Dim::Dim1, (&[3], &[3])),
            LayerDesc::new("combiner", LayerKind::Combiner, Dim::Dim1)
                .input(Dim::Dim1, (&[3], &[3]))
                .output(Dim::Dim1, &[3]),
            LayerDesc::new("output", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[3])
                .output(Dim::Dim1, &[1]),
        ]
    }
}

fn main() {
    let descs: Vec<_> = models::stage1()
        .into_iter()
        .chain(models::branch_a())
        .chain(models::branch_b())
        .chain(models::stage2())
        .collect();

    let plan = Plan::from_layer_descs(descs).expect("Invalid plan");
    let mut model = plan.build();

    // Инициализация параметров
    {
        let mut store = model.param_store().lock().unwrap();
        for i in 0..store.len() {
            store.set_param(i, rand::random::<f32>() * 0.01);
        }
    }

    let x = Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]);
    let target = Tensor2D::new(vec![vec![0.8]]);

    let sgd_desc = OptimizerDesc::new()
        .add(OptCubeDesc::ScaleGradient(0.01))
        .add(OptCubeDesc::ApplyUpdate);

    for epoch in 0..200 {
        let (pred, ctxs) = model.forward(DynamicTensor::Dim1(x.clone()));
        let loss_desc = LossDesc::from_chain(
            ElementChain::new().add(Box::new(Sub)).add(Box::new(Square)),
            Aggregation::Mean, 1, 1, 1,
        );
        let (loss, delta) = model.compute_loss(loss_desc, &pred, &DynamicTensor::Dim1(target.clone()));
        let (_, grads) = model.backward(&ctxs, delta);
        model.update_params(sgd_desc.clone(), &grads[0]);
        if epoch % 50 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
    }
}