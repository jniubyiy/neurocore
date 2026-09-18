// src/compute_manager/operators_v2/operator_v2.rs
//
// Контракт оператора v2.
//
// Оператор — единственная точка, куда распределитель передаёт job.
// Всё, что оператор делает дальше (потоки, Vulkan-очереди, миграции),
// снаружи невидимо.
//
// Жизненный цикл:
//
//   1. SmartDistributor выбирает оператора по `JobKind` и `TopologySnapshot`.
//   2. `let handle = operator.submit(job);` — оператор возвращает handle
//      сразу, не блокируя вызывающего. Внутри может быть:
//        * CPU: job уходит в WorkerPool / ControlThreadPool;
//        * GPU: job уходит в очередь GPU-треда;
//        * Memory: job исполняется синхронно (submit = исполнение),
//          но всё равно возвращает handle для единообразия.
//   3. `let result = operator.drain(handle);` — блокирует вызывающего до
//      готовности именно этого job'а. Возвращает `JobResult`.
//
// `try_drain(handle)` — необязательный неблокирующий опрос. Полезен для
// `dispatch_batch` (перекрытие submit/drain разных job'ов). По умолчанию
// реализуется через `drain`, но конкретные операторы могут переопределить.
//
// Трейт требует `Send + Sync + 'static`: `Arc<dyn OperatorV2>` живёт в
// `SmartDistributor` и вызывается из главного потока, из GPU-треда и т.д.
//
// Оператор НЕ знает:
//   * от какого сегмента / batch'а пришёл job;
//   * какой loss, какие слои, какой optimizer;
//   * кто его вызвал и что будет с результатом.
// Он видит только поля `Job` / `JobResult` из `jobs_v2::types`.

use crate::compute_manager::jobs_v2::{Job, JobHandle, JobResult, OperatorKind};

use super::capacity_v2::OperatorCapacity;

/// Единый контракт оператора v2 (Memory / CPU / GPU).
pub trait OperatorV2: Send + Sync + 'static {
    /// Идентификатор оператора. Должен совпадать с `JobHandle::operator`
    /// у всех handle'ов, выданных этим оператором.
    fn kind(&self) -> OperatorKind;

    /// Принимает job и возвращает handle немедленно.
    ///
    /// Гарантии:
    ///   * вызов не блокирует вызывающий поток на время исполнения job'а
    ///     (для CPU/GPU — асинхронно через собственные очереди; для Memory —
    ///     допускается синхронное исполнение, но возврат handle должен быть
    ///     быстрым);
    ///   * `handle.kind == job.kind()`;
    ///   * `handle.operator == self.kind()`;
    ///   * если очередь переполнена и политика оператора не предполагает
    ///     блокировки, вызов может паниковать — вызывающий должен проверять
    ///     `capacity().is_available()` перед `submit`.
    fn submit(&self, job: Job) -> JobHandle;

    /// Блокирует вызывающий поток до готовности результата `handle`.
    ///
    /// # Паника
    /// Паникует, если handle не принадлежит этому оператору или уже был
    /// забран предыдущим `drain` / `try_drain` (результат выдаётся один раз).
    fn drain(&self, handle: JobHandle) -> JobResult;

    /// Неблокирующий опрос.
    ///
    /// Возвращает `Some(result)` немедленно, если результат готов,
    /// `None` — если job ещё исполняется. Как и `drain`, «забирает»
    /// результат: повторный вызов вернёт панику.
    fn try_drain(&self, handle: JobHandle) -> Option<JobResult>;

    /// Снимок загрузки оператора.
    ///
    /// Используется `SmartDistributor::strategy` для решения, куда
    /// отправлять следующий job (в частности, когда есть выбор между
    /// CPU и GPU).
    fn capacity(&self) -> OperatorCapacity;
}