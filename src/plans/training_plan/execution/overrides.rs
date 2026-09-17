// src/plans/training_plan/execution/overrides.rs
//
// Layer-aware канонические инициализации.
//
// После того как пользовательский `Initializer` применён ко всем параметрам,
// для ряда слоёв применяются канонические инициализации. Это критично для
// слоёв, где произвольная инициализация (например, uniform[-0.1, 0.1])
// ломает семантику или загоняет слой в нежелательный режим.
//
// Для `BatchRenorm1d` каноническая инициализация:
//
//     γ = 1, β = 0, r = 1, d = 0
//
// При таком init выход слоя имеет нулевое среднее и единичную дисперсию по
// батчу (y ≈ x_hat), что сохраняет ~50% активных признаков после
// следующего ReLU и обеспечивает корректный поток градиента.
//
// Для `PerFeatureAttention` переопределяется только `self_bias_h` —
// обучаемая добавка к диагонали attention-ядра головы h:
//
//     self_bias_h ∈ [2.0, 4.0)  (детерминированно, от buffer_idx и start)
//
// Инициализация в [2.0, 4.0) выбрана по результатам двух экспериментов:
//
//   1. При sb ∈ [-0.1, 0.1] слой садится в режим чистого усреднения по
//      времени и не выходит оттуда за 300 эпох даже с lr = 0.05
//      (MSE = 0.039 ≈ MSE(mean_t(x))).
//
//   2. При sb ∈ [0.5, 1.5] слой уходит с плато усреднения, но остаётся
//      в промежуточном режиме (MSE = 0.036), всё ещё хуже identity
//      (0.029). Attention-ядро сглаживает сигнал сильнее, чем оптимально.
//
// Старт с sb ∈ [2.0, 4.0) — ближе к identity: при большом sb вклад
// `self_bias · v_t` в числитель доминирует над `φ(q_t) @ kv`, и выход
// головы почти не сглаживается. Из этой точки обучение может сползти
// к оптимуму (MSE ≈ 0.017 при α* ≈ 0.44), но не застревает в усреднении,
// потому что для этого пришлось бы сильно уменьшить sb — это требует
// существенно большего шага, чем для его увеличения.
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

                // ---------- PerFeatureAttention ----------
                //
                // Перезаписываем только позиции self_bias каждой головы.
                // Все остальные параметры остаются после основной
                // RandomUniform-инициализации.
                //
                // Раскладка параметров одной головы (см. per_feature_attention.rs):
                //     [0, 2·dh)              Wq (row-major)
                //     [2·dh, 3·dh)           bq
                //     [3·dh, 5·dh)           Wk
                //     [5·dh, 6·dh)           bk
                //     [6·dh, 8·dh)           Wv
                //     [8·dh, 9·dh)           bv
                //     [9·dh, 10·dh)          Wo
                //     [10·dh, 10·dh+1)       bo
                //     [10·dh+1, 10·dh+2)     self_bias
                if let Some(pfa) = layer.as_per_feature_attention() {
                    let d_model = pfa.d_model;
                    let d_head = pfa.d_head;
                    let head_pc = 10 * d_head + 2;

                    // Детерминированный xorshift64, засеянный от
                    // (buffer_idx, start). Это исключает зависимость от
                    // глобального RNG и делает результат воспроизводимым
                    // независимо от числа слоёв в модели.
                    let mut state: u64 = (slice.buffer_idx as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ (slice.start as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    if state == 0 {
                        state = 0xDEAD_BEEF_CAFE_BABE;
                    }

                    for h in 0..d_model {
                        // Один шаг xorshift64.
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;

                        // Берём 24 бита — этого достаточно для равномерного
                        // распределения на float без потери точности.
                        let unit = ((state & 0x00FF_FFFF) as f32) / 0x0100_0000u32 as f32;

                        // self_bias ∈ [2.0, 4.0).
                        let sb = 2.0 + 2.0 * unit;

                        let sb_pos = slice.start + h * head_pc + 10 * d_head + 1;
                        out.push((slice.buffer_idx, sb_pos, vec![sb]));
                    }
                }
            }
        }
    }

    out
}