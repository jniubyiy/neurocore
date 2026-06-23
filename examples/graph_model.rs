// examples/graph_model.rs

use neurocore::model_plan::{Plan, LayerDesc, LayerKind, Dim};

mod models {
    use neurocore::model_plan::{LayerDesc, LayerKind, Dim};

    pub fn model1() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("input", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[10usize])
                .output(Dim::Dim1, &[6usize]),

            LayerDesc::new("split_to_branches", LayerKind::Splitter, Dim::Dim1)
                .input(Dim::Dim1, &[6usize])
                .output(Dim::Dim1, (&[2usize], &[4usize])),

            LayerDesc::new("route_split", LayerKind::SplitterConnector, Dim::Dim1)
                .input(Dim::Dim1, (&[2usize], &[4usize])),
        ]
    }

    pub fn model2() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("route_a_in", LayerKind::SplitterConnector, Dim::Dim1)
                .output(Dim::Dim1, &[2usize]),

            LayerDesc::new("branch_a", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[2usize])
                .output(Dim::Dim1, &[3usize]),

            LayerDesc::new("route_a_out", LayerKind::CombinerConnector, Dim::Dim1)
                .input(Dim::Dim1, &[3usize]),
        ]
    }

    pub fn model3() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("route_b_in", LayerKind::SplitterConnector, Dim::Dim1)
                .output(Dim::Dim1, &[4usize]),

            LayerDesc::new("branch_b", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[4usize])
                .output(Dim::Dim1, &[5usize]),

            LayerDesc::new("route_b_out", LayerKind::CombinerConnector, Dim::Dim1)
                .input(Dim::Dim1, &[5usize]),
        ]
    }

    pub fn model4() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("route_combine_in", LayerKind::CombinerConnector, Dim::Dim1)
                .output(Dim::Dim1, (&[3usize], &[5usize])),

            LayerDesc::new("combine_from_branches", LayerKind::Combiner, Dim::Dim1)
                .input(Dim::Dim1, (&[3usize], &[5usize]))
                .output(Dim::Dim1, &[8usize]),

            LayerDesc::new("output", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[8usize])
                .output(Dim::Dim1, &[2usize]),
        ]
    }
}

fn main() {
    let descs: Vec<_> = models::model1()
        .into_iter()
        .chain(models::model2())
        .chain(models::model3())
        .chain(models::model4())
        .collect();

    match Plan::from_descs(descs) {
        Ok(plan) => {
            println!("✅ План успешно создан!");
            println!("   Количество слоёв: {}", 12);
            let _model = plan.build();
            println!("   Модель успешно собрана.");
        }
        Err(e) => {
            println!("❌ Ошибка создания плана: {}", e);
        }
    }
}