// src/compute_manager/distributor_v2/strategy.rs
//
// Правила выбора оператора под конкретный JobKind.
//
// Стратегия — чистая функция: принимает ссылку на job и снимок топологии,
// возвращает идентификатор оператора. Никаких side-effect'ов, никаких
// обращений к операторам, никаких блокировок.
//
// Таблица решений:
//
//   JobKind                      | Operator | Условие
//   -----------------------------|----------|-------------------------------------
//   Migrate                      | Memory   | всегда
//   ForwardSegment               | GPU      | has_gpu && segment_param_count > HEAVY_THRESHOLD
//   ForwardSegment               | CPU      | иначе
//   BackwardSegment              | GPU      | как ForwardSegment
//   BackwardSegment              | CPU      | иначе
//   Loss                         | GPU      | has_gpu && pred на GPU && target на GPU
//   Loss                         | CPU      | иначе
//   OptimizerModifyGrads         | CPU      | всегда (hybrid через gpu_compute)
//   OptimizerApplyUpdate         | CPU      | всегда (hybrid через gpu_compute)
//   DimOp                        | CPU      | всегда
//   ConnectorOp                  | CPU      | всегда
//   ParamInit                    | CPU      | всегда
//
// Порог «тяжести» сегмента согласован со старым
// `compute_manager::compute_executor::placement` (HEAVY_THRESHOLD = 1000):
// до этого порога overhead запуска GPU-операции превышает выигрыш от
// переноса мелких сегментов в VRAM.

use crate::compute_manager::jobs_v2::{Job, JobKind, OperatorKind};
use crate::layers::UniversalLayer;

use super::topology::TopologySnapshot;

/// Порог «тяжести» сегмента по суммарному числу параметров.
pub const HEAVY_THRESHOLD: usize = 1000;

/// Возвращает оператора, которому следует передать job.
pub fn select_operator(job: &Job, snapshot: &TopologySnapshot) -> OperatorKind {
    match job.kind() {
        JobKind::Migrate => OperatorKind::Memory,

        JobKind::ForwardSegment | JobKind::BackwardSegment => {
            match layers_of(job) {
                Some(layers) => select_compute(layers, snapshot),
                None => OperatorKind::Cpu, // защита от рассинхрона job.kind ↔ payload
            }
        }

        JobKind::Loss => select_loss(job, snapshot),

        // Обе фазы оптимизатора — CPU-код (кубики из plans::optimizer_plan::cube).
        // Если буферы физически лежат на GPU, `CpuOperatorV2` внутри
        // `OptimizerExpr::*_hybrid` скачает их, шагнёт и зальёт обратно.
        // Ссылка на `GpuCompute` вкладывается в job заранее (prepare_job).
        //
        // Фаза 1 и фаза 2 — независимые job'ы, каждый выбирается CPU-оператором
        // отдельно. Между ними в графе вызывается `adapter_pass` (Фаза 4 плана).
        JobKind::OptimizerModifyGrads => OperatorKind::Cpu,
        JobKind::OptimizerApplyUpdate => OperatorKind::Cpu,

        // DimOp и ConnectorOp — memory-bound операции.
        JobKind::DimOp => OperatorKind::Cpu,
        JobKind::ConnectorOp => OperatorKind::Cpu,

        // Инициализация параметров — CPU-код (StdRng + запись в буфер).
        JobKind::ParamInit => OperatorKind::Cpu,
    }
}

// ---------------------------------------------------------------------------
// Вспомогательные функции
// ---------------------------------------------------------------------------

/// Извлекает слои сегмента из ForwardSegmentJob / BackwardSegmentJob.
fn layers_of(job: &Job) -> Option<&[Box<dyn UniversalLayer>]> {
    match job {
        Job::ForwardSegment(j) => Some(j.layers.as_slice()),
        Job::BackwardSegment(j) => Some(j.layers.as_slice()),
        _ => None,
    }
}

/// Выбирает вычислительное устройство для сегмента.
fn select_compute(
    layers: &[Box<dyn UniversalLayer>],
    snapshot: &TopologySnapshot,
) -> OperatorKind {
    if !snapshot.has_gpu {
        return OperatorKind::Cpu;
    }

    let total_params: usize = layers.iter().map(|l| l.param_len()).sum();
    if total_params > HEAVY_THRESHOLD {
        OperatorKind::Gpu
    } else {
        OperatorKind::Cpu
    }
}

/// Выбирает устройство для Loss.
///
/// GPU выбирается только если:
///   * GPU доступен в топологии;
///   * pred и target уже лежат в VRAM (иначе GpuOperatorV2 вернёт Failed).
fn select_loss(job: &Job, snapshot: &TopologySnapshot) -> OperatorKind {
    if !snapshot.has_gpu {
        return OperatorKind::Cpu;
    }
    match job {
        Job::Loss(l) => {
            if l.pred.is_gpu() && l.target.is_gpu() {
                OperatorKind::Gpu
            } else {
                OperatorKind::Cpu
            }
        }
        _ => OperatorKind::Cpu,
    }
}