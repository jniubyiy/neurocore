// examples/graph_model.rs

use neurocore::model_plan::{Plan, LayerDesc, LayerKind, Dim};

fn main() {
    // Модель с разветвлением: один вход -> линейный слой -> разделитель на два выхода
    // Каждая ветвь проходит свою обработку, затем объединитель сливает их обратно.
    let descs = vec![
        // Общий входной слой
        LayerDesc::new("input", LayerKind::Linear, Dim::Dim1)
            .input(Dim::Dim1, &[10])
            .output(Dim::Dim1, &[6]),

        // Разделитель: превращает 6 в два выхода размерами 2 и 4
        LayerDesc::splitter_connector("split", Dim::Dim1, 6, vec![2, 4]),

        // Ветвь A (работает с размером 2)
        LayerDesc::new("branch_a", LayerKind::Linear, Dim::Dim1)
            .input(Dim::Dim1, &[2])
            .output(Dim::Dim1, &[3]),

        // Ветвь B (работает с размером 4)
        LayerDesc::new("branch_b", LayerKind::Linear, Dim::Dim1)
            .input(Dim::Dim1, &[4])
            .output(Dim::Dim1, &[5]),

        // Объединитель: сливает 3 и 5 в один выход размера 8
        LayerDesc::combiner_connector("combine", Dim::Dim1, vec![3, 5]),

        // Завершающий слой
        LayerDesc::new("output", LayerKind::Linear, Dim::Dim1)
            .input(Dim::Dim1, &[8])
            .output(Dim::Dim1, &[2]),
    ];

    // Попытка создать план – он должен проходить проверку размерностей
    match Plan::from_descs(descs) {
        Ok(plan) => {
            println!("✅ План успешно создан!");
            println!("   Количество слоёв: {}", 6); // можно вывести plan.blueprints.len()
            // При попытке сборки возникнет ожидаемая ошибка
            match plan.build() {
                Ok(_) => println!("Модель собрана (неожиданно)"),
                Err(e) => println!("⚠️  Ожидаемая ошибка сборки: {}", e),
            }
        }
        Err(e) => {
            println!("❌ Ошибка создания плана: {}", e);
        }
    }
}