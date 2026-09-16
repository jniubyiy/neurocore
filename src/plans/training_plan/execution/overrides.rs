// src/plans/training_plan/execution/overrides.rs
//
// Layer-aware канонические инициализации.
//
// После того как пользовательский `Initializer` применён ко всем параметрам,
// для ряда слоёв применяются канонические инициализации. Это критично для
// слоёв нормализации, где произвольная инициализация (например,
// `uniform[-0.1, 0.1]`) ломает семантику слоя.
//
// В частности, для `BatchRenorm1d` канонической является:
//
//     γ = 1, β = 0, r = 1, d = 0
//
// При таком init выход слоя имеет нулевое среднее и единичную дисперсию по
// батчу (y ≈ x_hat), что сохраняет ~50% активных признаков после
// следующего ReLU и обеспечивает корректный поток градиента.
//
// При `γ, r ~ 0.05` (типичный uniform[-0.1, 0.1]) вклад x_hat в y
// становится ~ 0.0025·x_hat, доминирует β со случайным знаком, часть
// признаков получает y < 0 ВСЕГДА, ReLU их жёстко обнуляет, и градиент
// через них не проходит. Это приводит к dead-ReLU по признакам и
// остановке обучения.
//
// Функция возвращает список (buffer_idx, offset, values), которые нужно
// записать в ParamStore поверх пользовательской инициализации.

use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::Model;

pub(super) fn build_layer_aware_overrides(
    model: &MixedModel,
) -> Vec<(usize, usize, Vec<f32>)> {
    let mut out = Vec::new();

    for m in model.models() {
        if let Model::UniversalProcessor(layers, slices, _) = m {
            for (i, layer) in layers.iter().enumerate() {
                let slice = &slices[i];

                // ---------- BatchRenorm1d ----------
                if let Some(br) = layer.as_batch_renorm() {
                    let f = br.features;
                    // Раскладка: [γ (f) | β (f) | r (f) | d (f)]
                    let mut buf = vec![0.0f32; 4 * f];
                    for c in 0..f {
                        buf[0 * f + c] = 1.0; // γ
                        // buf[1*f + c] = 0.0;   // β
                        buf[2 * f + c] = 1.0; // r
                        // buf[3*f + c] = 0.0;   // d
                    }
                    out.push((slice.buffer_idx, slice.start, buf));
                }
            }
        }
    }

    out
}