// src/logging/training_monitor.rs

use std::collections::VecDeque;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// Конфигурация монитора обучения.
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Размер окна для анализа тренда loss (количество последних шагов).
    pub window_size: usize,
    /// Порог относительного увеличения loss, после которого выдаётся предупреждение.
    pub loss_increase_threshold: f32,
    /// Коэффициент замедления: если loss уменьшается менее чем на X% за эпоху, LR мал.
    pub slowdown_threshold: f32,
    /// Максимальное количество эпох без улучшения перед предупреждением.
    pub patience: usize,
    /// Сохранять ли дампы NaN.
    pub dump_on_nan: bool,
    /// Максимальное количество предупреждений одного типа за всё обучение (0 – без ограничений)
    pub max_warnings_per_type: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            loss_increase_threshold: 0.05,
            slowdown_threshold: 0.001,
            patience: 5,
            dump_on_nan: true,
            max_warnings_per_type: 3,   // ограничиваем повторяющиеся сообщения
        }
    }
}

/// Предупреждение, генерируемое монитором.
#[derive(Debug, Clone)]
pub enum Warning {
    LossIncrease {
        current: f32,
        previous: f32,
        suggestion: String,
    },
    LossStagnation {
        current: f32,
        suggestion: String,
    },
    GradientExplosion {
        norm: f32,
    },
    NaNValue {
        location: String,
    },
}

/// Сводка за одну эпоху.
#[derive(Debug, Clone)]
pub struct EpochSummary {
    pub epoch: usize,
    pub avg_loss: f32,
    pub loss_change: Option<f32>,
    pub warnings: Vec<Warning>,
    pub nan_detected: bool,
}

/// Итоговая сводка после обучения.
#[derive(Debug, Clone)]
pub struct TrainingSummary {
    pub epochs: usize,
    pub final_loss: f32,
    pub warnings: Vec<Warning>,
    pub nan_count: usize,
}

/// Монитор обучения. Отслеживает потери, градиенты, NaN и даёт рекомендации.
pub struct TrainingMonitor {
    config: MonitorConfig,
    loss_history: VecDeque<f32>,
    step_in_epoch: usize,
    epoch_loss_sum: f32,
    nan_steps: usize,
    epoch: usize,
    prev_epoch_loss: Option<f32>,
    current_lr: f32,
    dump_dir: PathBuf,
    enabled: bool,
    patience_counter: usize,
    best_loss: f32,
    all_warnings: Vec<Warning>,
    // счётчики уже выведенных предупреждений по типам (упрощённо: строковое представление типа)
    warning_type_counts: std::collections::HashMap<String, usize>,
    /// Флаг, что в текущей эпохе уже обнаружен NaN (чтобы не генерировать лишние предупреждения)
    nan_in_this_epoch: bool,
}

impl TrainingMonitor {
    pub fn new(config: MonitorConfig, learning_rate: f32, dump_dir: PathBuf) -> Self {
        if config.dump_on_nan {
            let _ = fs::create_dir_all(&dump_dir);
        }
        let window_size = config.window_size;
        Self {
            config,
            loss_history: VecDeque::with_capacity(window_size),
            step_in_epoch: 0,
            epoch_loss_sum: 0.0,
            nan_steps: 0,
            epoch: 0,
            prev_epoch_loss: None,
            current_lr: learning_rate,
            dump_dir,
            enabled: true,
            patience_counter: 0,
            best_loss: f32::MAX,
            all_warnings: Vec::new(),
            warning_type_counts: std::collections::HashMap::new(),
            nan_in_this_epoch: false,
        }
    }

    pub fn disabled() -> Self {
        Self {
            config: MonitorConfig::default(),
            loss_history: VecDeque::new(),
            step_in_epoch: 0,
            epoch_loss_sum: 0.0,
            nan_steps: 0,
            epoch: 0,
            prev_epoch_loss: None,
            current_lr: 0.0,
            dump_dir: PathBuf::new(),
            enabled: false,
            patience_counter: 0,
            best_loss: f32::MAX,
            all_warnings: Vec::new(),
            warning_type_counts: std::collections::HashMap::new(),
            nan_in_this_epoch: false,
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Зафиксировать один шаг обучения.
    pub fn record_step(&mut self, loss: f32, grads: Option<&[f32]>, params: Option<&[f32]>) {
        if !self.enabled {
            return;
        }

        self.step_in_epoch += 1;
        self.epoch_loss_sum += loss;

        if self.loss_history.len() >= self.config.window_size {
            self.loss_history.pop_front();
        }
        self.loss_history.push_back(loss);

        // Проверка NaN/Inf.
        if loss.is_nan() || loss.is_infinite() {
            self.nan_steps += 1;
            if !self.nan_in_this_epoch {
                let location = format!("epoch {}, step {}", self.epoch, self.step_in_epoch);
                let warning = Warning::NaNValue {
                    location: location.clone(),
                };
                self.add_warning(warning);
                eprintln!("[TrainingMonitor] NaN/Inf detected at {}! loss = {:?}", location, loss);
                if self.config.dump_on_nan {
                    self.dump_nan_state(&location, loss, grads, params);
                }
                self.nan_in_this_epoch = true; // только одно сообщение об NaN за эпоху
            }
        }

        // Проверка градиентов (если переданы).
        if let Some(grads) = grads {
            let norm = grads.iter().map(|&g| g as f64 * g as f64).sum::<f64>().sqrt() as f32;
            if norm > 1e6 {
                let warning = Warning::GradientExplosion { norm };
                self.add_warning(warning);
            }
        }
    }

    /// Завершить эпоху и получить сводку.
    pub fn end_epoch(&mut self) -> EpochSummary {
        if !self.enabled {
            return EpochSummary {
                epoch: self.epoch,
                avg_loss: 0.0,
                loss_change: None,
                warnings: vec![],
                nan_detected: false,
            };
        }

        let avg_loss = if self.step_in_epoch > 0 {
            self.epoch_loss_sum / self.step_in_epoch as f32
        } else {
            0.0
        };

        let loss_change = self.prev_epoch_loss.map(|prev| avg_loss - prev);

        let mut warnings = Vec::new();

        // Анализ скорости обучения и стагнации – только если нет NaN в этой или предыдущей эпохе
        if !self.nan_in_this_epoch && !self.prev_epoch_loss.map(|p| p.is_nan()).unwrap_or(false) {
            warnings = self.analyze_learning_rate(avg_loss);
        }

        // Проверка patience.
        if avg_loss.is_nan() || avg_loss.is_infinite() {
            // не обновляем best_loss, когда значения сломаны
        } else if avg_loss < self.best_loss {
            self.best_loss = avg_loss;
            self.patience_counter = 0;
        } else {
            self.patience_counter += 1;
            if self.patience_counter >= self.config.patience {
                let w = Warning::LossStagnation {
                    current: avg_loss,
                    suggestion: format!(
                        "No improvement for {} epochs. Best loss = {:.6}. Consider reducing LR or changing model.",
                        self.patience_counter, self.best_loss
                    ),
                };
                warnings.push(w);
            }
        }

        // Фильтруем дублирующиеся предупреждения по лимиту
        let filtered_warnings = self.filter_warnings(warnings);

        // Сброс счётчиков шага и флага NaN для следующей эпохи
        self.step_in_epoch = 0;
        self.epoch_loss_sum = 0.0;
        self.prev_epoch_loss = Some(avg_loss);
        self.epoch += 1;
        self.nan_in_this_epoch = false;

        EpochSummary {
            epoch: self.epoch - 1,
            avg_loss,
            loss_change,
            warnings: filtered_warnings,
            nan_detected: self.nan_steps > 0,
        }
    }

    /// Получить итоговую сводку.
    pub fn summary(&self) -> TrainingSummary {
        TrainingSummary {
            epochs: self.epoch,
            final_loss: self.prev_epoch_loss.unwrap_or(0.0),
            warnings: self.all_warnings.clone(),
            nan_count: self.nan_steps,
        }
    }

    // ----------------------------------------------------------------
    // Приватные методы
    // ----------------------------------------------------------------

    /// Добавляет предупреждение в общий список, соблюдая лимит на тип.
    fn add_warning(&mut self, warning: Warning) {
        let type_key = match &warning {
            Warning::LossIncrease { .. } => "LossIncrease",
            Warning::LossStagnation { .. } => "LossStagnation",
            Warning::GradientExplosion { .. } => "GradientExplosion",
            Warning::NaNValue { .. } => "NaNValue",
        };
        let count = self.warning_type_counts.entry(type_key.to_string()).or_insert(0);
        if self.config.max_warnings_per_type == 0 || *count < self.config.max_warnings_per_type {
            self.all_warnings.push(warning);
            *count += 1;
        }
    }

    /// Фильтрует список предупреждений, оставляя только те, которые ещё не превысили лимит.
    fn filter_warnings(&mut self, warnings: Vec<Warning>) -> Vec<Warning> {
        let mut out = Vec::new();
        for w in warnings {
            let type_key = match &w {
                Warning::LossIncrease { .. } => "LossIncrease",
                Warning::LossStagnation { .. } => "LossStagnation",
                Warning::GradientExplosion { .. } => "GradientExplosion",
                Warning::NaNValue { .. } => "NaNValue",
            };
            let count = self.warning_type_counts.entry(type_key.to_string()).or_insert(0);
            if self.config.max_warnings_per_type == 0 || *count < self.config.max_warnings_per_type {
                out.push(w.clone());
                self.all_warnings.push(w);
                *count += 1;
            }
        }
        out
    }

    /// Анализирует динамику loss и возвращает предупреждения.
    fn analyze_learning_rate(&self, avg_loss: f32) -> Vec<Warning> {
        let mut warnings = Vec::new();

        if let Some(prev) = self.prev_epoch_loss {
            if prev.is_nan() || avg_loss.is_nan() {
                return warnings; // нечего анализировать
            }

            let change = avg_loss - prev;
            let rel_change = if prev.abs() > 1e-8 {
                change / prev.abs()
            } else {
                0.0
            };

            if rel_change > self.config.loss_increase_threshold {
                warnings.push(Warning::LossIncrease {
                    current: avg_loss,
                    previous: prev,
                    suggestion: format!(
                        "Loss increased by {:.2}% ({} -> {}). Consider reducing learning rate (current: {:.2e}).",
                        rel_change * 100.0,
                        prev,
                        avg_loss,
                        self.current_lr
                    ),
                });
            } else if rel_change.abs() < self.config.slowdown_threshold && avg_loss > 1e-6 {
                warnings.push(Warning::LossStagnation {
                    current: avg_loss,
                    suggestion: format!(
                        "Loss is stagnating (change {:.2}%). Consider increasing learning rate or changing model capacity.",
                        rel_change * 100.0
                    ),
                });
            }

            // Высокая дисперсия в окне – подавляем, если уже много таких предупреждений
            if self.loss_history.len() >= 2 {
                let mean: f32 = self.loss_history.iter().sum::<f32>() / self.loss_history.len() as f32;
                let variance: f32 = self
                    .loss_history
                    .iter()
                    .map(|&l| {
                        let d = l - mean;
                        d * d
                    })
                    .sum::<f32>()
                    / self.loss_history.len() as f32;
                let std_dev = variance.sqrt();
                if std_dev > mean * 0.5 && mean > 1e-6 {
                    warnings.push(Warning::LossIncrease {
                        current: avg_loss,
                        previous: prev,
                        suggestion: format!(
                            "High loss variance detected (std={:.6}, mean={:.6}). Gradient may be unstable. Consider reducing learning rate.",
                            std_dev, mean
                        ),
                    });
                }
            }
        }

        warnings
    }

    /// Сохраняет дамп состояния при обнаружении NaN.
    fn dump_nan_state(&self, location: &str, loss: f32, grads: Option<&[f32]>, params: Option<&[f32]>) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let filename = format!(
            "nan_dump_epoch{}_step{}_{}.txt",
            self.epoch,
            self.step_in_epoch,
            timestamp
        );
        let path = self.dump_dir.join(&filename);

        if let Ok(mut file) = fs::File::create(&path) {
            writeln!(file, "NaN detected at: {}", location).ok();
            writeln!(file, "Loss: {}", loss).ok();
            if let Some(g) = grads {
                let norm: f64 = g.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
                writeln!(file, "Gradient norm: {:.6}", norm).ok();
                writeln!(file, "Gradient sample (first 10): {:?}", &g[..g.len().min(10)]).ok();
            }
            if let Some(p) = params {
                writeln!(file, "Parameter sample (first 10): {:?}", &p[..p.len().min(10)]).ok();
            }
            eprintln!("[TrainingMonitor] NaN state dumped to {}", path.display());
        } else {
            eprintln!("[TrainingMonitor] Failed to create dump file at {}", path.display());
        }
    }
}