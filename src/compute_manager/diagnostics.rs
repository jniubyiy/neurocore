// src/compute_manager/diagnostics.rs

use std::collections::HashMap;
use std::fmt::Write;
use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::memory_executor::TensorBufferId;
use crate::compute_manager::graph::types::Segment;

/// Статистика по параметрам модели или градиентам
#[derive(Debug, Clone)]
pub struct ParamsStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std_dev: f32,
    /// L2-норма вектора градиентов (если переданы)
    pub gradient_norm: f32,
}

/// Диагностическая информация об одном тензоре (буфере)
#[derive(Debug, Clone)]
pub struct TensorDiagInfo {
    pub size_elements: usize,
    pub device: DeviceId,
    pub handle: TensorBufferId,
}

/// Полный контекст ошибки, собираемый во время выполнения
#[derive(Debug, Clone)]
pub struct DiagContext {
    /// Список зарегистрированных устройств (их строковые описания)
    pub device_info: Vec<String>,
    /// Краткое описание сегментов модели
    pub segment_summary: Vec<String>,
    /// Снимок состояния тензоров (TensorBufferId -> информация)
    pub tensor_snapshot: HashMap<TensorBufferId, TensorDiagInfo>,
    /// Статистика параметров модели
    pub params_stats: Option<ParamsStats>,
    /// Описание последнего выполнявшегося шага
    pub last_step_description: String,
    /// Текст ошибки
    pub error_message: String,
}

impl DiagContext {
    /// Создаёт минимальный контекст с сообщением об ошибке
    pub fn new(error_message: impl Into<String>) -> Self {
        Self {
            device_info: Vec::new(),
            segment_summary: Vec::new(),
            tensor_snapshot: HashMap::new(),
            params_stats: None,
            last_step_description: String::new(),
            error_message: error_message.into(),
        }
    }
}

/// Собирает диагностический контекст из переданных компонентов.
///
/// # Аргументы
/// * `device_info` – список строк с описанием устройств
/// * `segment_summary` – список строк с описанием сегментов
/// * `_segments` – ссылка на сегменты (задел на будущее)
/// * `tensor_loc` – карта расположения тензоров (TensorBufferId -> DeviceId)
/// * `tensor_sizes` – карта размеров тензоров (TensorBufferId -> количество элементов)
/// * `params` – плоский срез всех параметров модели
/// * `grads` – плоский срез градиентов (может быть пустым)
/// * `last_step_description` – описание шага, на котором произошла ошибка
/// * `error_message` – текст ошибки
pub fn capture_diagnostics(
    device_info: Vec<String>,
    segment_summary: Vec<String>,
    _segments: &[Segment],               // задел на будущее
    tensor_loc: &HashMap<TensorBufferId, DeviceId>,
    tensor_sizes: &HashMap<TensorBufferId, usize>,
    params: &[f32],
    grads: &[f32],
    last_step_description: String,
    error_message: String,
) -> DiagContext {
    let mut tensor_snapshot = HashMap::new();
    for (&id, &device) in tensor_loc {
        let size_elements = tensor_sizes.get(&id).copied().unwrap_or(0);
        tensor_snapshot.insert(
            id,
            TensorDiagInfo {
                size_elements,
                device,
                handle: id,
            },
        );
    }

    let params_stats = if !params.is_empty() {
        Some(compute_params_stats(params, grads))
    } else {
        None
    };

    DiagContext {
        device_info,
        segment_summary,
        tensor_snapshot,
        params_stats,
        last_step_description,
        error_message,
    }
}

/// Вычисляет статистику по параметрам и градиентам.
///
/// * `params` – срез параметров
/// * `grads` – срез градиентов (может быть пустым для пропуска нормы)
pub fn compute_params_stats(params: &[f32], grads: &[f32]) -> ParamsStats {
    let n = params.len();
    if n == 0 {
        return ParamsStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std_dev: 0.0,
            gradient_norm: 0.0,
        };
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &v in params {
        min = min.min(v);
        max = max.max(v);
        sum += v as f64;
    }
    let mean = (sum / n as f64) as f32;

    let mut sq_diff = 0.0f64;
    for &v in params {
        let d = v as f64 - mean as f64;
        sq_diff += d * d;
    }
    let variance = sq_diff / n.max(1) as f64;
    let std_dev = variance.sqrt() as f32;

    let gradient_norm = if grads.len() == n {
        let sum_sq: f64 = grads.iter().map(|&g| (g as f64) * (g as f64)).sum();
        sum_sq.sqrt() as f32
    } else {
        0.0
    };

    ParamsStats {
        min,
        max,
        mean,
        std_dev,
        gradient_norm,
    }
}

/// Форматирует `DiagContext` в подробный текстовый отчёт.
pub fn format_diagnostics_report(ctx: &DiagContext) -> String {
    let mut s = String::new();

    writeln!(s, "=============== NEUROCORE DIAGNOSTICS ===============").ok();
    writeln!(s, "Error: {}", ctx.error_message).ok();
    writeln!(s, "Last step: {}", ctx.last_step_description).ok();

    writeln!(s, "\n--- Devices ---").ok();
    for (i, dev) in ctx.device_info.iter().enumerate() {
        writeln!(s, "  [{}] {}", i, dev).ok();
    }

    writeln!(s, "\n--- Segments ---").ok();
    for (i, seg) in ctx.segment_summary.iter().enumerate() {
        writeln!(s, "  [{}] {}", i, seg).ok();
    }

    writeln!(s, "\n--- Tensor Snapshot ---").ok();
    if ctx.tensor_snapshot.is_empty() {
        writeln!(s, "  (empty)").ok();
    } else {
        for (&id, info) in &ctx.tensor_snapshot {
            writeln!(
                s,
                "  id {}: size={} elems, device={:?}, handle={}",
                id.0, info.size_elements, info.device, info.handle.0
            )
            .ok();
        }
    }

    if let Some(ref stats) = ctx.params_stats {
        writeln!(s, "\n--- Parameter Stats ---").ok();
        writeln!(s, "  min = {}", stats.min).ok();
        writeln!(s, "  max = {}", stats.max).ok();
        writeln!(s, "  mean = {}", stats.mean).ok();
        writeln!(s, "  std_dev = {}", stats.std_dev).ok();
        writeln!(s, "  gradient L2 norm = {}", stats.gradient_norm).ok();
    }

    writeln!(s, "======================================================").ok();
    s
}