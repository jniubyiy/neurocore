// src/layers/adapter/stats.rs

//! Диагностика работы градиентных адаптеров (MIGRATION_PLAN.md §7, Фаза 6).
//!
//! # Назначение
//!
//! Собрать метрики о том, как адаптеры влияют на градиент слоя. Центральная
//! метрика — **scale** = `after_l2 / before_l2` (изменение L2-нормы
//! собственного среза градиента слоя).
//!
//!   * `scale == 1.0` — адаптер ничего не сделал (no-op, baseline);
//!   * `scale > 1.0` — адаптер усилил градиент;
//!   * `scale < 1.0` — адаптер ослабил градиент.
//!
//! # Стоимость замеров
//!
//! L2-норма вычисляется на CPU. Для GPU-буфера это означает скачивание
//! данных — дорого. Поэтому замеры выполняются **только** при активной
//! диагностике (`NEUROCORE_DEBUG_ADAPTER=1`). Без флага
//! `before_l2 = after_l2 = NaN`, а `scale()` возвращает `0.0`.
//!
//! Это гарантирует, что в обычном режиме работы Фаза 6 не влияет на
//! производительность обучения.

use std::collections::HashMap;

/// Статистика одного вызова `GradientAdapter::apply`.
///
/// Соответствует одной паре «(сегмент, слой) → адаптер» на одном шаге.
#[derive(Debug, Clone)]
pub struct AdapterCallStats {
    /// Человекочитаемое имя адаптера (`adapter.name()`).
    pub name: &'static str,

    /// L2-норма собственного среза градиента до `apply`.
    /// `NaN`, если замер не выполнялся (нет `NEUROCORE_DEBUG_ADAPTER=1`).
    pub before_l2: f32,

    /// L2-норма собственного среза градиента после `apply`.
    /// `NaN`, если замер не выполнялся.
    pub after_l2: f32,
}

impl AdapterCallStats {
    /// Отношение `after_l2 / before_l2`.
    ///
    /// Возвращает `0.0`, если:
    ///   * замер не выполнялся (`before_l2`/`after_l2` = NaN или ∞);
    ///   * `before_l2` пренебрежимо мала (< 1e-30).
    #[inline]
    pub fn scale(&self) -> f32 {
        if !self.before_l2.is_finite() || !self.after_l2.is_finite() {
            return 0.0;
        }
        if self.before_l2.abs() < 1e-30 {
            return 0.0;
        }
        self.after_l2 / self.before_l2
    }
}

/// Статистика одного `GraphV2::adapter_pass`.
///
/// Содержит по одной записи на каждый вызов `apply` за шаг обучения.
/// Для многослойных сегментов записей столько же, сколько слоёв с
/// адаптерами.
#[derive(Debug, Clone, Default)]
pub struct AdapterPassStats {
    pub calls: Vec<AdapterCallStats>,
}

impl AdapterPassStats {
    /// `true`, если ни один адаптер не вызывался.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Число вызовов адаптеров за этот pass.
    #[inline]
    pub fn len(&self) -> usize {
        self.calls.len()
    }
}

/// Агрегат по одному имени адаптера за всё обучение.
#[derive(Debug, Clone)]
pub struct AdapterAggregate {
    /// Сколько раз вызывался этот адаптер.
    pub calls: usize,

    /// Сколько из этих вызовов реально измерялись (L2 посчитана).
    pub measured_calls: usize,

    /// Среднее значение `scale` по измеренным вызовам.
    pub mean_scale: f32,

    /// Минимальный `scale` по измеренным вызовам.
    pub min_scale: f32,

    /// Максимальный `scale` по измеренным вызовам.
    pub max_scale: f32,
}

impl Default for AdapterAggregate {
    fn default() -> Self {
        Self {
            calls: 0,
            measured_calls: 0,
            mean_scale: 0.0,
            min_scale: f32::INFINITY,
            max_scale: f32::NEG_INFINITY,
        }
    }
}

/// Итоговая сводка работы адаптеров за всё обучение.
///
/// Пишется в `TrainingResult::adapter_summary` (Фаза 6 плана).
/// `Option` на уровне `TrainingResult`: `Some`, если был хотя бы один
/// вызов адаптера за обучение; иначе `None`.
#[derive(Debug, Clone, Default)]
pub struct AdapterSummary {
    /// Всего вызовов адаптеров за обучение (по всем слоям, шагам, эпохам).
    pub total_calls: usize,

    /// Из них реально измеренных (с посчитанной L2).
    pub total_measured: usize,

    /// Статистика по каждому имени адаптера.
    pub per_adapter: HashMap<String, AdapterAggregate>,
}

impl AdapterSummary {
    /// Добавляет статистику одного `adapter_pass`.
    ///
    /// Вызывается из `execute_v2.rs` после каждого `graph.adapter_pass()`.
    pub fn absorb(&mut self, pass: &AdapterPassStats) {
        for call in &pass.calls {
            self.total_calls += 1;

            let agg = self
                .per_adapter
                .entry(call.name.to_string())
                .or_default();
            agg.calls += 1;

            // Учитываем только валидные замеры.
            let s = call.scale();
            if s.is_finite() && s > 0.0
                && call.before_l2.is_finite()
                && call.after_l2.is_finite()
                && call.before_l2.abs() > 1e-30
            {
                agg.measured_calls += 1;
                let n = agg.measured_calls as f32;
                // Инкрементальное среднее (Уэлфорд-подобное).
                agg.mean_scale += (s - agg.mean_scale) / n;
                if s < agg.min_scale {
                    agg.min_scale = s;
                }
                if s > agg.max_scale {
                    agg.max_scale = s;
                }
                self.total_measured += 1;
            }
        }
    }

    /// Форматирует сводку в человекочитаемый вид (для логов).
    pub fn report(&self) -> String {
        if self.total_calls == 0 {
            return "AdapterSummary: no adapters were called".to_string();
        }

        let mut s = String::new();
        s.push_str(&format!(
            "AdapterSummary: total_calls={}, total_measured={}\n",
            self.total_calls, self.total_measured
        ));

        // Стабильный порядок вывода: сортируем имена.
        let mut names: Vec<&String> = self.per_adapter.keys().collect();
        names.sort();

        for name in names {
            let agg = &self.per_adapter[name];
            if agg.measured_calls > 0 {
                s.push_str(&format!(
                    "  {}: calls={} measured={} \
                     mean_scale={:.4} min_scale={:.4} max_scale={:.4}\n",
                    name,
                    agg.calls,
                    agg.measured_calls,
                    agg.mean_scale,
                    agg.min_scale,
                    agg.max_scale
                ));
            } else {
                s.push_str(&format!(
                    "  {}: calls={} (L2 not measured; \
                     set NEUROCORE_DEBUG_ADAPTER=1 to enable)\n",
                    name, agg.calls
                ));
            }
        }

        s
    }
}

// ============================================================================
// Юнит-тесты
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_summary_report() {
        let s = AdapterSummary::default();
        assert_eq!(s.report(), "AdapterSummary: no adapters were called");
    }

    #[test]
    fn absorb_counts_calls() {
        let mut summary = AdapterSummary::default();

        let pass = AdapterPassStats {
            calls: vec![
                AdapterCallStats {
                    name: "linear",
                    before_l2: f32::NAN,
                    after_l2: f32::NAN,
                },
                AdapterCallStats {
                    name: "linear",
                    before_l2: f32::NAN,
                    after_l2: f32::NAN,
                },
            ],
        };
        summary.absorb(&pass);

        assert_eq!(summary.total_calls, 2);
        assert_eq!(summary.total_measured, 0);
        let agg = summary.per_adapter.get("linear").unwrap();
        assert_eq!(agg.calls, 2);
        assert_eq!(agg.measured_calls, 0);
    }

    #[test]
    fn absorb_measures_scales() {
        let mut summary = AdapterSummary::default();

        // pass 1: scale = 2.0 (before=1.0, after=2.0)
        let pass1 = AdapterPassStats {
            calls: vec![AdapterCallStats {
                name: "test",
                before_l2: 1.0,
                after_l2: 2.0,
            }],
        };
        summary.absorb(&pass1);

        // pass 2: scale = 4.0 (before=1.0, after=4.0)
        let pass2 = AdapterPassStats {
            calls: vec![AdapterCallStats {
                name: "test",
                before_l2: 1.0,
                after_l2: 4.0,
            }],
        };
        summary.absorb(&pass2);

        assert_eq!(summary.total_calls, 2);
        assert_eq!(summary.total_measured, 2);

        let agg = summary.per_adapter.get("test").unwrap();
        assert_eq!(agg.calls, 2);
        assert_eq!(agg.measured_calls, 2);
        assert!((agg.mean_scale - 3.0).abs() < 1e-6, "mean should be 3.0, got {}", agg.mean_scale);
        assert!((agg.min_scale - 2.0).abs() < 1e-6);
        assert!((agg.max_scale - 4.0).abs() < 1e-6);
    }

    #[test]
    fn scale_returns_zero_for_unmeasured() {
        let call = AdapterCallStats {
            name: "x",
            before_l2: f32::NAN,
            after_l2: f32::NAN,
        };
        assert_eq!(call.scale(), 0.0);

        let call = AdapterCallStats {
            name: "x",
            before_l2: 0.0,
            after_l2: 1.0,
        };
        assert_eq!(call.scale(), 0.0);

        let call = AdapterCallStats {
            name: "x",
            before_l2: 1.0,
            after_l2: 1.0,
        };
        assert_eq!(call.scale(), 1.0);
    }
}