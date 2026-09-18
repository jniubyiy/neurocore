// src/compute_manager/operators_v2/gpu_v2/queue_v2.rs
//
// Выделенный GPU-тред и очередь заданий для GpuOperatorV2.
//
// Зачем отдельный тред:
//   * Vulkan-операции требуют большого стека (32 МБ) — Windows-поток
//     по умолчанию имеет 1 МБ, что мало для глубоких графов;
//   * единая точка сериализации всех GPU-команд — Vulkan-очередь внутри
//     `GpuCompute` защищена `queue_lock`, но сам job (несколько
//     `run_compute_shader` подряд) эффективнее исполнять в одном
//     выделенном треде, не переключая контекст.
//
// Поток живёт пока не придёт `Shutdown`. `Drop for GpuQueueV2` посылает
// `Shutdown` и join'ит поток.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread;

use crate::compute_manager::gpu::GpuCompute;
use crate::compute_manager::jobs_v2::{Job, JobResult};
use crate::compute_manager::memory_executor::MemoryExecutor;

const GPU_THREAD_STACK_SIZE: usize = 32 * 1024 * 1024;

// ============================================================================
// Сообщения очереди
// ============================================================================

/// Сообщение для GPU-треда.
pub enum GpuTaskMessage {
    /// Задание на исполнение. `id` — локальный идентификатор в слоте
    /// результатов. Тред запишет `JobResult` в `slots[id]` и уведомит `cv`.
    Job {
        id: u64,
        job: Job,
        slots: Arc<Mutex<HashMap<u64, JobResult>>>,
        cv: Arc<Condvar>,
    },
    /// Завершение потока.
    Shutdown,
}

// ============================================================================
// Очередь
// ============================================================================

/// Очередь заданий + выделенный GPU-тред.
pub struct GpuQueueV2 {
    tx: std::sync::mpsc::Sender<GpuTaskMessage>,
    thread: Option<thread::JoinHandle<()>>,
}

impl GpuQueueV2 {
    /// Запускает GPU-тред. Тред обрабатывает сообщения до `Shutdown`.
    pub fn new(
        gpu: Arc<GpuCompute>,
        memory: Arc<RwLock<MemoryExecutor>>,
    ) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<GpuTaskMessage>();

        let thread = thread::Builder::new()
            .name("neurocore-gpu-v2".to_string())
            .stack_size(GPU_THREAD_STACK_SIZE)
            .spawn(move || {
                gpu_thread_loop(rx, gpu, memory);
            })
            .expect("GpuQueueV2: failed to spawn GPU thread");

        Self {
            tx,
            thread: Some(thread),
        }
    }

    /// Отправляет сообщение в очередь. Не блокирует.
    ///
    /// Если тред уже остановлен — молча игнорирует ошибку отправки.
    pub fn send(&self, msg: GpuTaskMessage) {
        let _ = self.tx.send(msg);
    }
}

impl Drop for GpuQueueV2 {
    fn drop(&mut self) {
        let _ = self.tx.send(GpuTaskMessage::Shutdown);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

// ============================================================================
// Цикл GPU-треда
// ============================================================================

fn gpu_thread_loop(
    rx: std::sync::mpsc::Receiver<GpuTaskMessage>,
    gpu: Arc<GpuCompute>,
    _memory: Arc<RwLock<MemoryExecutor>>,
) {
    while let Ok(msg) = rx.recv() {
        match msg {
            GpuTaskMessage::Job { id, job, slots, cv } => {
                let result = execute_gpu_job(&gpu, job);
                {
                    let mut s = slots.lock().unwrap();
                    s.insert(id, result);
                }
                cv.notify_all();
            }
            GpuTaskMessage::Shutdown => break,
        }
    }
}

// ============================================================================
// Исполнение одного GPU-job'а
// ============================================================================

fn execute_gpu_job(
    gpu: &Arc<GpuCompute>,
    job: Job,
) -> JobResult {
    use crate::compute_manager::gpu::processor::{
        process_backward_gpu_buffered, process_forward_gpu_buffered,
    };
    use crate::compute_manager::jobs_v2::ForwardContextsV2;
    use crate::loss_plan::compute_loss_gpu_buffered_handle;

    match job {
        Job::ForwardSegment(f) => {
            let params_view = f.params.clone();
            // params — MatrixBufferHandle; для GPU-сегмента параметры уже в VRAM.
            let (output, ctxs) = process_forward_gpu_buffered(
                gpu.as_ref(),
                f.layers.as_slice(),
                f.slices.as_slice(),
                &params_view,
                f.input,
            );
            JobResult::Forward {
                output,
                contexts: ForwardContextsV2::Sequential(ctxs),
            }
        }
        Job::BackwardSegment(b) => {
            let ctxs = b.contexts.first_chunk();
            let params_view = b.params.clone();
            let grad_params_view = b.grad_params.clone();
            let grad_input = process_backward_gpu_buffered(
                gpu.as_ref(),
                b.layers.as_slice(),
                b.slices.as_slice(),
                ctxs.as_slice(),
                &params_view,
                b.grad_output,
                &grad_params_view,
            );
            JobResult::Backward { grad_input }
        }
        Job::Loss(l) => {
            if !l.pred.is_gpu() || !l.target.is_gpu() {
                return JobResult::Failed(
                    "GpuOperatorV2::loss: pred or target is not on GPU".into(),
                );
            }
            let (value, grad_pred) = compute_loss_gpu_buffered_handle(
                gpu.as_ref(),
                &l.expr,
                &l.pred,
                &l.target,
            );
            JobResult::Loss { value, grad_pred }
        }
        other => JobResult::Failed(format!(
            "GpuOperatorV2: unsupported job kind {:?}",
            other.kind()
        )),
    }
}