// src/compute_manager/graph_v2/bridge_v2.rs
//
// Мост между v2-графом и каноническими per-layer инициализациями
// из старого пути.
//
// Задача: после того как `GraphV2::init_params` применит generic
// Initializer ко всем параметрам, для ряда слоёв нужно переопределить
// значения на канонические, потому что generic-init ломает их семантику:
//
//   * `BatchRenorm1d` — каноническая инициализация γ=1, β=0, r=1, d=0.
//     При uniform[-0.1, 0.1] выход слоя имел бы случайный масштаб и
//     смещение, что лишает слой смысла.
//
//   * `PerFeatureAttention` — каноническое `self_bias_h ∈ [2.0, 4.0)`
//     (детерминированно, от `buffer_idx` и `slice.start`). При малом
//     `self_bias` слой садится в режим чистого усреднения по времени и
//     не выходит оттуда. Диапазон [2.0, 4.0) выбран экспериментально
//     (см. `plans/training_plan/execution/overrides.rs`, v1).
//
//   * `LearnableMish` — каноническое λ = 1.0. При generic
//     uniform[-0.1, 0.1] λ ≈ 0, слой вырожден в почти-нулевую функцию:
//     tanh(λ·sp) ≈ λ·sp ≈ 0 ⇒ y ≈ 0. Гипотеза подтверждена логами
//     (см. NEUROCORE_DEBUG_LMISH=1): λ ≈ -0.048, x.l2 ≈ 0.84,
//     y.l2 ≈ 0.028, gi.l2 ослаблен в ~30 раз относительно go.l2.
//     При λ = 1.0 слой сразу ведёт себя как классический Mish.
//
//   * `AdaptiveNormalization` — канонические γ=1 для всех трёх ветвей
//     (LayerNorm/RMSNorm/BatchNorm), β=0, логиты=0. При generic
//     uniform[-0.1, 0.1] γ ≈ 0, слой ослабляет сигнал в ~20 раз:
//     x.l2 ≈ 10.0 → y.l2 ≈ 0.54, go.l2 ≈ 5.4e-3 → gi.l2 ≈ 2.7e-4.
//     Гипотеза подтверждена логами (см. NEUROCORE_DEBUG_ADNORM=1):
//     mean|γ_ln| ≈ 6.4e-4, mean|γ_rms| ≈ 2.8e-3, mean|γ_bn| ≈ 3.7e-2.
//     При γ=1 слой ведёт себя как комбинированная нормализация
//     с равным вкладом всех трёх ветвей (softmax(0,0,0) = 1/3).
//
// Подробное обоснование первых двух случаев — в v1-файле `overrides.rs`.
// Здесь — тонкий адаптер, чтобы не менять замороженный v1-код.
//
// Файл ничего не вычисляет сам и не хранит состояния. Чистая функция
// от `&GraphV2` к списку значений, которые нужно записать поверх
// generic-инициализации.

use crate::layers::UniversalLayer;

use super::model_v2::GraphV2;
use super::types_v2::SegmentKindV2;

/// Возвращает список `(buffer_idx, start, values)`, которые нужно
/// записать в `ParamStore` поверх generic-инициализации.
///
/// Работает только с сегментами `SegmentKindV2::Universal` — остальные
/// виды сегментов (DimOp, Connector) канонических параметров не имеют.
pub(crate) fn build_layer_aware_overrides_v2(
    graph: &GraphV2,
) -> Vec<(usize, usize, Vec<f32>)> {
    let mut out = Vec::new();

    for seg in &graph.segments {
        let SegmentKindV2::Universal { layers, slices } = &seg.kind else {
            continue;
        };

        for (i, layer) in layers.iter().enumerate() {
            let Some(slice) = slices.get(i) else {
                // Согласованность slices и layers гарантируется builder'ом.
                // Если когда-то нарушится, пропускаем — generic-init
                // останется в силе, никаких паник.
                continue;
            };

            // ---------- BatchRenorm1d ----------
            //
            // Раскладка параметров: [γ (f) | β (f) | r (f) | d (f)].
            // Канонические значения: γ=1, β=0, r=1, d=0.
            if let Some(br) = layer.as_batch_renorm() {
                let f = br.features;
                let mut buf = vec![0.0f32; 4 * f];
                for c in 0..f {
                    buf[0 * f + c] = 1.0; // γ
                    // buf[1*f + c] = 0.0;   // β (оставляем 0)
                    buf[2 * f + c] = 1.0; // r
                    // buf[3*f + c] = 0.0;   // d (оставляем 0)
                }
                out.push((slice.buffer_idx, slice.start, buf));
            }

            // ---------- PerFeatureAttention ----------
            //
            // Раскладка параметров одной головы:
            //     [0, 2·dh)          Wq (row-major)
            //     [2·dh, 3·dh)       bq
            //     [3·dh, 5·dh)       Wk
            //     [5·dh, 6·dh)       bk
            //     [6·dh, 8·dh)       Wv
            //     [8·dh, 9·dh)       bv
            //     [9·dh, 10·dh)      Wo
            //     [10·dh, 10·dh+1)   bo
            //     [10·dh+1, 10·dh+2) self_bias
            //
            // Перезаписываем только self_bias каждой головы.
            if let Some(pfa) = layer.as_per_feature_attention() {
                let d_model = pfa.d_model;
                let d_head = pfa.d_head;
                let head_pc = 10 * d_head + 2;

                // Детерминированный xorshift64. Засеян от (buffer_idx, start),
                // чтобы результат не зависел от глобального RNG и был
                // воспроизводим независимо от числа слоёв в модели.
                let mut state: u64 = (slice.buffer_idx as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (slice.start as u64)
                        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
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
                    let unit =
                        ((state & 0x00FF_FFFF) as f32) / 0x0100_0000u32 as f32;

                    // self_bias ∈ [2.0, 4.0).
                    let sb = 2.0 + 2.0 * unit;

                    let sb_pos = slice.start + h * head_pc + 10 * d_head + 1;
                    out.push((slice.buffer_idx, sb_pos, vec![sb]));
                }
            }

            // ---------- LearnableMish ----------
            //
            // Раскладка параметров: [λ] — один скаляр на слой.
            // Каноническая инициализация: λ = 1.0 (классический Mish).
            //
            // При generic RandomUniform[-0.1, 0.1] λ попадает в диапазон
            // около нуля, слой вырождается в почти-нулевую функцию:
            //   tanh(λ·sp) ≈ λ·sp ≈ 0  ⇒  y ≈ 0.
            // Это подтверждено логами NEUROCORE_DEBUG_LMISH=1.
            // При λ = 1.0 слой сразу ведёт себя как стандартный Mish.
            if layer.as_learnable_mish().is_some() {
                out.push((slice.buffer_idx, slice.start, vec![1.0f32]));
            }

            // ---------- AdaptiveNormalization ----------
            //
            // Раскладка параметров (7·f):
            //     [0·f, 1·f)   γ_ln
            //     [1·f, 2·f)   β_ln
            //     [2·f, 3·f)   γ_rms
            //     [3·f, 4·f)   γ_bn
            //     [4·f, 5·f)   β_bn
            //     [5·f, 6·f)   logits_ln
            //     [6·f, 7·f)   logits_rms
            //
            // Канонические значения:
            //     γ_ln = 1.0, β_ln = 0.0,
            //     γ_rms = 1.0,
            //     γ_bn = 1.0, β_bn = 0.0,
            //     logits_ln = 0.0, logits_rms = 0.0.
            //
            // При softmax(0, 0, 0) все три ветви получают вес 1/3 —
            // равный вклад LayerNorm / RMSNorm / BatchNorm. Обучение
            // дальше может сдвинуть логиты в нужную сторону.
            //
            // При generic RandomUniform[-0.1, 0.1] γ ≈ 0, слой ослабляет
            // сигнал в ~20 раз (x.l2 ≈ 10.0 → y.l2 ≈ 0.54), и все
            // градиенты через слой тоже ослабляются в ~20 раз. Это
            // подтверждено логами NEUROCORE_DEBUG_ADNORM=1.
            if let Some(adnorm) = layer.as_adaptive_normalization() {
                let f = adnorm.features;
                let mut buf = vec![0.0f32; 7 * f];
                for c in 0..f {
                    buf[0 * f + c] = 1.0; // γ_ln
                    // buf[1*f + c] = 0.0;   // β_ln
                    buf[2 * f + c] = 1.0; // γ_rms
                    buf[3 * f + c] = 1.0; // γ_bn
                    // buf[4*f + c] = 0.0;   // β_bn
                    // buf[5*f + c] = 0.0;   // logits_ln
                    // buf[6*f + c] = 0.0;   // logits_rms
                }
                out.push((slice.buffer_idx, slice.start, buf));
            }
        }
    }

    out
}