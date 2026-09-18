// src/compute_manager/graph_v2/observer_v2.rs
//
// GraphObserverV2 — наблюдатель за процессом обучения v2.
//
// Наблюдатель не участвует в вычислениях: он получает уже посчитанные
// числа (loss, нормы градиента, нормы параметров) и делает из них выводы.
// Граф вызывает его в трёх точках:
//
//   * `record_step(loss, grad_norm)` — после каждого optimizer_step;
//   * `record_param_norm(norm)`      — раз в конце эпохи (после агрегации);
//   * `end_epoch() -> EpochReportV2` — на границе эпохи; возвращает отчёт.
//
// Дополнительно:
//   * `record_nan()` — если в шаге обнаружены NaN/Inf (граф сам решает,
//     как их детектировать, и сообщает наблюдателю факт);
//   * `should_reassign()` — вызывается графом перед новой эпохой; если
//     `true`, граф дёргает `distributor.on_epoch_boundary(...)`.
//
// Наблюдатель НЕ знает:
//   * про граф, сегменты, слои, распределитель, операторы;
//   * про optimizer или learning rate;
//   * про то, какие именно параметры обучаются.
// Он видит только поток скалярных метрик.

use std::collections::VecDeque;

// ============================================================================
// Конфигурация
// ============================================================================

/// Пороги и окна наблюдателя.
#[derive(Debug, Clone)]
pub struct MonitorConfigV2 {
    /// Размер окна для шаговой истории loss (для тренда).
    pub step_window: usize,

    /// Размер окна для шаговой истории grad_norm.
    pub grad_window: usize,

    /// Размер окна для эпоховой истории loss (для плато).
    pub epoch_window: usize,

    /// Порог относительного роста loss за эпоху, выше которого
    /// выдаётся предупреждение и `should_reassign` возвращает `true`.
    pub loss_increase_threshold: f32,

    /// Порог относительного изменения loss за эпоху, ниже которого
    /// эпоха считается «без прогресса» (стагнация).
    pub slowdown_threshold: f32,

    /// Сколько подряд эпох без улучшения `best_loss` считать плато.
    pub plateau_patience: usize,

    /// Порог L2-нормы градиента, выше которого выдаётся GradientExplosion.
    pub grad_explosion_threshold: f32,

    /// Минимальное число эпох между перераспределениями.
    /// Гарантирует, что граф не будет дёргать distributor каждый шаг.
    pub min_epochs_between_reassign: usize,

    /// Максимум предупреждений одного типа (0 — без лимита).
    pub max_warnings_per_type: usize,
}

impl Default for MonitorConfigV2 {
    fn default() -> Self {
        Self {
            step_window: 100,
            grad_window: 100,
            epoch_window: 20,
            loss_increase_threshold: 0.05,
            slowdown_threshold: 0.001,
            plateau_patience: 5,
            grad_explosion_threshold: 1e6,
            min_epochs_between_reassign: 3,
            max_warnings_per_type: 3,
        }
    }
}

// ============================================================================
// Предупреждения
// ============================================================================

/// Предупреждение, сформированное наблюдателем.
#[derive(Debug, Clone)]
pub enum WarningV2 {
    /// Loss за эпоху вырос по сравнению с предыдущей.
    LossIncrease {
        current: f32,
        previous: f32,
        rel_change: f32,
    },
    /// Loss за эпоху почти не изменился.
    LossStagnation {
        current: f32,
        rel_change: f32,
    },
    /// L2-норма градиента превысила порог.
    GradientExplosion { norm: f32 },
    /// В одном из шагов эпохи обнаружены NaN/Inf.
    NanValue { epoch: usize, step: usize },
}

impl WarningV2 {
    /// Строковый ключ для учёта «сколько раз этот тип уже выдан».
    fn type_key(&self) -> &'static str {
        match self {
            WarningV2::LossIncrease { .. } => "LossIncrease",
            WarningV2::LossStagnation { .. } => "LossStagnation",
            WarningV2::GradientExplosion { .. } => "GradientExplosion",
            WarningV2::NanValue { .. } => "NanValue",
        }
    }
}

// ============================================================================
// Отчёт эпохи
// ============================================================================

/// Результат наблюдения за одну эпоху. Возвращается графу из `end_epoch`.
#[derive(Debug, Clone)]
pub struct EpochReportV2 {
    /// Номер эпохи (0-indexed).
    pub epoch: usize,

    /// Средний loss по шагам эпохи.
    pub avg_loss: f32,

    /// Изменение avg_loss относительно предыдущей эпохи
    /// (`None` для первой эпохи).
    pub loss_change: Option<f32>,

    /// Средняя L2-норма градиента по шагам эпохи, если граф передавал
    /// `grad_norm` в `record_step` (`None` — если ни разу не передал).
    pub grad_norm_avg: Option<f32>,

    /// Максимальная L2-норма градиента по шагам эпохи.
    pub grad_norm_max: Option<f32>,

    /// Норма параметров на конец эпохи, если передана.
    pub param_norm: Option<f32>,

    /// Количество NaN/Inf-событий в эпохе.
    pub nan_count: usize,

    /// Предупреждения эпохи.
    pub warnings: Vec<WarningV2>,

    /// Текстовые рекомендации (для логирования графом).
    pub recommendations: Vec<String>,
}

// ============================================================================
// Направление тренда
// ============================================================================

/// Направление тренда loss за окно.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossTrendV2 {
    /// Loss убывает.
    Decreasing,
    /// Loss не меняется значимо.
    Stable,
    /// Loss растёт.
    Increasing,
    /// Недостаточно данных.
    Unknown,
}

// ============================================================================
// Наблюдатель
// ============================================================================

/// Наблюдатель за процессом обучения v2.
pub struct GraphObserverV2 {
    config: MonitorConfigV2,

    /// Текущая эпоха.
    epoch: usize,

    /// Шаг внутри текущей эпохи.
    step_in_epoch: usize,

    /// Шаговая история loss.
    loss_history: VecDeque<f32>,

    /// Шаговая история L2-нормы градиента.
    grad_norm_history: VecDeque<f32>,

    /// Эпоховая история avg_loss.
    epoch_losses: VecDeque<f32>,

    /// Эпоховая история L2-нормы параметров.
    param_norm_history: VecDeque<f32>,

    /// Сумма loss за текущую эпоху.
    epoch_loss_sum: f64,

    /// Кол-во шагов в текущей эпохе (для avg).
    epoch_step_count: usize,

    /// Сумма grad_norm за текущую эпоху.
    epoch_grad_sum: f64,

    /// Максимальная grad_norm за текущую эпоху.
    epoch_grad_max: f32,

    /// Сколько раз граф сообщал grad_norm за эпоху.
    epoch_grad_count: usize,

    /// NaN-события в текущей эпохе.
    epoch_nan_count: usize,

    /// Общее число NaN-событий за всё время.
    total_nan_count: usize,

    /// Последнее значение grad_norm (для выявления GradientExplosion).
    last_grad_norm: Option<f32>,

    /// Лучший avg_loss за всё время.
    best_loss: f32,

    /// Сколько эпох подряд лучший avg_loss не улучшался.
    plateau_counter: usize,

    /// Последняя эпоха, на которой вызывался `on_epoch_boundary`
    /// (используется `should_reassign`).
    last_reassign_epoch: usize,

    /// Накопленные предупреждения за всё время.
    all_warnings: Vec<WarningV2>,

    /// Счётчики предупреждений по типам.
    warning_type_counts: std::collections::HashMap<&'static str, usize>,
}

impl GraphObserverV2 {
    /// Создаёт наблюдателя с конфигурацией по умолчанию.
    pub fn new() -> Self {
        Self::with_config(MonitorConfigV2::default())
    }

    /// Создаёт наблюдателя с явной конфигурацией.
    pub fn with_config(config: MonitorConfigV2) -> Self {
        Self {
            config,
            epoch: 0,
            step_in_epoch: 0,
            loss_history: VecDeque::new(),
            grad_norm_history: VecDeque::new(),
            epoch_losses: VecDeque::new(),
            param_norm_history: VecDeque::new(),
            epoch_loss_sum: 0.0,
            epoch_step_count: 0,
            epoch_grad_sum: 0.0,
            epoch_grad_max: f32::NEG_INFINITY,
            epoch_grad_count: 0,
            epoch_nan_count: 0,
            total_nan_count: 0,
            last_grad_norm: None,
            best_loss: f32::MAX,
            plateau_counter: 0,
            last_reassign_epoch: 0,
            all_warnings: Vec::new(),
            warning_type_counts: std::collections::HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Запись
    // -----------------------------------------------------------------------

    /// Записать один шаг.
    ///
    /// `loss` — значение функции потерь на этом шаге.
    /// `grad_norm` — L2-норма градиента (если граф её считает; иначе `None`).
    pub fn record_step(&mut self, loss: f32, grad_norm: Option<f32>) {
        // Шаговая история loss.
        push_bounded(&mut self.loss_history, loss, self.config.step_window);

        // Шаговая история grad_norm + обновления эпохи.
        if let Some(norm) = grad_norm {
            push_bounded(&mut self.grad_norm_history, norm, self.config.grad_window);
            self.epoch_grad_sum += norm as f64;
            self.epoch_grad_count += 1;
            if norm > self.epoch_grad_max {
                self.epoch_grad_max = norm;
            }
            self.last_grad_norm = Some(norm);

            if norm > self.config.grad_explosion_threshold {
                self.add_warning(WarningV2::GradientExplosion { norm });
            }
        }

        // Аккумуляторы эпохи.
        self.epoch_loss_sum += loss as f64;
        self.epoch_step_count += 1;
        self.step_in_epoch += 1;

        // NaN / Inf в loss.
        if loss.is_nan() || loss.is_infinite() {
            self.record_nan();
        }
    }

    /// Сообщить о NaN/Inf в текущем шаге.
    ///
    /// Может вызываться как из `record_step` (при обнаружении), так и
    /// отдельно — если граф находит NaN в другом месте (например, в
    /// градиентах, но не в loss).
    pub fn record_nan(&mut self) {
        self.epoch_nan_count += 1;
        self.total_nan_count += 1;
        let w = WarningV2::NanValue {
            epoch: self.epoch,
            step: self.step_in_epoch,
        };
        self.add_warning(w);
    }

    /// Записать L2-норму параметров на конец эпохи.
    ///
    /// Граф вызывает это один раз перед `end_epoch` (в противном случае
    /// `param_norm` в отчёте будет `None`).
    pub fn record_param_norm(&mut self, norm: f32) {
        push_bounded(
            &mut self.param_norm_history,
            norm,
            self.config.epoch_window,
        );
    }

    // -----------------------------------------------------------------------
    // Завершение эпохи
    // -----------------------------------------------------------------------

    /// Завершает эпоху, возвращает отчёт и продвигает внутренние счётчики.
    pub fn end_epoch(&mut self) -> EpochReportV2 {
        let avg_loss = if self.epoch_step_count > 0 {
            (self.epoch_loss_sum / self.epoch_step_count as f64) as f32
        } else {
            0.0
        };

        let prev_epoch_loss = self.epoch_losses.back().copied();
        let loss_change = prev_epoch_loss.map(|prev| avg_loss - prev);

        // История эпох.
        push_bounded(
            &mut self.epoch_losses,
            avg_loss,
            self.config.epoch_window,
        );

        // Best_loss и плато.
        if avg_loss.is_finite() && avg_loss < self.best_loss {
            self.best_loss = avg_loss;
            self.plateau_counter = 0;
        } else if self.epoch_step_count > 0 {
            self.plateau_counter += 1;
        }

        // Warnings эпохи.
        let mut warnings: Vec<WarningV2> = Vec::new();

        if let Some(prev) = prev_epoch_loss {
            if prev.is_finite() && avg_loss.is_finite() && prev.abs() > 1e-8 {
                let rel_change = (avg_loss - prev) / prev.abs();
                if rel_change > self.config.loss_increase_threshold {
                    warnings.push(WarningV2::LossIncrease {
                        current: avg_loss,
                        previous: prev,
                        rel_change,
                    });
                } else if rel_change.abs() < self.config.slowdown_threshold
                    && avg_loss > 1e-6
                {
                    warnings.push(WarningV2::LossStagnation {
                        current: avg_loss,
                        rel_change,
                    });
                }
            }
        }

        if self.plateau_counter >= self.config.plateau_patience {
            warnings.push(WarningV2::LossStagnation {
                current: avg_loss,
                rel_change: 0.0,
            });
        }

        // Записываем с учётом лимита.
        warnings = self.filter_warnings(warnings);

        // Рекомендации.
        let recommendations = build_recommendations(&warnings, self.plateau_counter);

        // Агрегаты grad_norm эпохи.
        let grad_norm_avg = if self.epoch_grad_count > 0 {
            Some((self.epoch_grad_sum / self.epoch_grad_count as f64) as f32)
        } else {
            None
        };
        let grad_norm_max = if self.epoch_grad_count > 0 {
            Some(self.epoch_grad_max)
        } else {
            None
        };

        // Param norm — последнее записанное.
        let param_norm = self.param_norm_history.back().copied();

        let report = EpochReportV2 {
            epoch: self.epoch,
            avg_loss,
            loss_change,
            grad_norm_avg,
            grad_norm_max,
            param_norm,
            nan_count: self.epoch_nan_count,
            warnings,
            recommendations,
        };

        // Сброс per-epoch счётчиков, инкремент эпохи.
        self.epoch_loss_sum = 0.0;
        self.epoch_step_count = 0;
        self.epoch_grad_sum = 0.0;
        self.epoch_grad_max = f32::NEG_INFINITY;
        self.epoch_grad_count = 0;
        self.epoch_nan_count = 0;
        self.step_in_epoch = 0;
        self.epoch += 1;

        report
    }

    // -----------------------------------------------------------------------
    // Принятие решений
    // -----------------------------------------------------------------------

    /// Возвращает `true`, если графу имеет смысл запросить
    /// перераспределение у `SmartDistributor::on_epoch_boundary`.
    ///
    /// Срабатывает при любом из условий:
    ///   * с последнего reassign прошло ≥ `min_epochs_between_reassign`;
    ///   * loss на последней эпохе вырос выше `loss_increase_threshold`;
    ///   * `plateau_counter >= plateau_patience`.
    pub fn should_reassign(&self) -> bool {
        let epochs_since = self.epoch.saturating_sub(self.last_reassign_epoch);
        if epochs_since < self.config.min_epochs_between_reassign {
            return false;
        }
        if self.is_plateau() {
            return true;
        }
        if let (Some(cur), Some(prev)) =
            (self.epoch_losses.back().copied(), second_last(&self.epoch_losses))
        {
            if prev.abs() > 1e-8 {
                let rel = (cur - prev) / prev.abs();
                if rel > self.config.loss_increase_threshold {
                    return true;
                }
            }
        }
        // Регулярный ре-assign каждые `min_epochs_between_reassign` эпох —
        // чтобы не застрять на одном плане.
        epochs_since >= self.config.min_epochs_between_reassign * 2
    }

    /// Отмечает, что перераспределение выполнено на текущей эпохе.
    pub fn mark_reassigned(&mut self) {
        self.last_reassign_epoch = self.epoch;
    }

    /// Плато: слишком много эпох без улучшения `best_loss`.
    pub fn is_plateau(&self) -> bool {
        self.plateau_counter >= self.config.plateau_patience
    }

    /// Дивергенция: значимый рост loss за последнюю эпоху.
    pub fn is_diverging(&self) -> bool {
        let (Some(cur), Some(prev)) =
            (self.epoch_losses.back().copied(), second_last(&self.epoch_losses))
        else {
            return false;
        };
        if !cur.is_finite() || !prev.is_finite() || prev.abs() < 1e-8 {
            return false;
        }
        (cur - prev) / prev.abs() > self.config.loss_increase_threshold
    }

    /// Текущее направление тренда loss по шаговой истории.
    pub fn loss_trend(&self) -> LossTrendV2 {
        if self.loss_history.len() < 2 {
            return LossTrendV2::Unknown;
        }
        let n = self.loss_history.len();
        // Сравниваем среднее первой и второй половины окна.
        let half = n / 2;
        let first_avg: f32 =
            self.loss_history.iter().take(half).sum::<f32>() / half as f32;
        let second_avg: f32 =
            self.loss_history.iter().skip(n - half).sum::<f32>() / half as f32;

        if first_avg.abs() < 1e-8 {
            return LossTrendV2::Unknown;
        }
        let rel = (second_avg - first_avg) / first_avg.abs();
        if rel < -self.config.slowdown_threshold {
            LossTrendV2::Decreasing
        } else if rel > self.config.slowdown_threshold {
            LossTrendV2::Increasing
        } else {
            LossTrendV2::Stable
        }
    }

    // -----------------------------------------------------------------------
    // Аксессоры
    // -----------------------------------------------------------------------

    /// Текущая эпоха (0-indexed).
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Лучший avg_loss за всё время.
    pub fn best_loss(&self) -> f32 {
        self.best_loss
    }

    /// Общее число NaN-событий за всё время.
    pub fn total_nan_count(&self) -> usize {
        self.total_nan_count
    }

    /// Все накопленные предупреждения.
    pub fn all_warnings(&self) -> &[WarningV2] {
        &self.all_warnings
    }

    /// История avg_loss по эпохам (окно).
    pub fn epoch_losses(&self) -> &VecDeque<f32> {
        &self.epoch_losses
    }

    // -----------------------------------------------------------------------
    // Внутреннее
    // -----------------------------------------------------------------------

    fn add_warning(&mut self, w: WarningV2) {
        let key = w.type_key();
        let count = self.warning_type_counts.entry(key).or_insert(0);
        if self.config.max_warnings_per_type == 0
            || *count < self.config.max_warnings_per_type
        {
            self.all_warnings.push(w);
            *count += 1;
        }
    }

    fn filter_warnings(&mut self, incoming: Vec<WarningV2>) -> Vec<WarningV2> {
        let mut out = Vec::new();
        for w in incoming {
            let key = w.type_key();
            let count = self.warning_type_counts.entry(key).or_insert(0);
            if self.config.max_warnings_per_type == 0
                || *count < self.config.max_warnings_per_type
            {
                self.all_warnings.push(w.clone());
                *count += 1;
                out.push(w);
            }
        }
        out
    }
}

impl Default for GraphObserverV2 {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Вспомогательные функции
// ============================================================================

fn push_bounded(q: &mut VecDeque<f32>, v: f32, cap: usize) {
    if cap == 0 {
        return;
    }
    while q.len() >= cap {
        q.pop_front();
    }
    q.push_back(v);
}

fn second_last(q: &VecDeque<f32>) -> Option<f32> {
    let n = q.len();
    if n < 2 {
        return None;
    }
    q.get(n - 2).copied()
}

fn build_recommendations(warnings: &[WarningV2], plateau_counter: usize) -> Vec<String> {
    let mut recs = Vec::new();
    for w in warnings {
        match w {
            WarningV2::LossIncrease { rel_change, .. } => {
                recs.push(format!(
                    "Loss increased by {:.2}%. Consider reducing LR or \
                     inspecting the data.",
                    rel_change * 100.0
                ));
            }
            WarningV2::LossStagnation { .. } => {
                recs.push(
                    "Loss stagnating. Consider increasing LR, changing \
                     optimizer, or adding model capacity."
                        .to_string(),
                );
            }
            WarningV2::GradientExplosion { norm } => {
                recs.push(format!(
                    "Gradient norm {:.3e} is very large. Apply gradient \
                     clipping or reduce LR.",
                    norm
                ));
            }
            WarningV2::NanValue { epoch, step } => {
                recs.push(format!(
                    "NaN/Inf detected at epoch {} step {}. Check LR, \
                     initialization, or loss scaling.",
                    epoch, step
                ));
            }
        }
    }
    if plateau_counter > 0 && recs.is_empty() {
        recs.push(format!(
            "No best_loss improvement for {} epochs. Consider \
             reassignment or LR adjustment.",
            plateau_counter
        ));
    }
    recs
}