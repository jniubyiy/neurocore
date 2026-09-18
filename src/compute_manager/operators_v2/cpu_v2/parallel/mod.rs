// src/compute_manager/cpu/parallel/mod.rs
//
// Параллельный forward/backward по чанкам батча.
//
// Ключевая идея forward'а:
//   * Планировщик разбивает батч на чанки (start, size, end).
//   * Для каждого чанка строится ChunkSlice — описание того, какие
//     строки входа он читает и в какие строки выхода он пишет свой
//     результат. В стандартном режиме in_range == out_range.
//   * Каждый воркер для каждого своего чанка:
//       - извлекает входной срез из общего input;
//       - прогоняет через всю цепочку слоёв;
//       - получает BufferedContext от каждого слоя (слой сам строит
//         свой контекст, включая per-chunk state, если он есть);
//       - пишет финальный результат в СВОЙ per-chunk буфер;
//       - сохраняет контексты слоёв в ctx_storage[chunk_id].
//   * После завершения всех воркеров выполняется фаза merge:
//     последовательно копирует per-chunk буферы в output по
//     out_start..out_end из плана.
//
// Backward устроен проще: градиенты пишутся в общий grad_input сразу
// по диапазону входного среза, потому что направление потока
// градиента однозначно определено forward'ом.
//
// Модуль разбит по функциональным блокам:
//   * `plan`         — типы плана чанкования;
//   * `tracker`      — учёт состояния чанков (для отладки);
//   * `shared`       — разделяемые между воркерами структуры данных;
//   * `chunk_ops`    — операции копирования чанков;
//   * `dims`         — определение размерностей слоёв;
//   * `dispatch`     — диспетчеризация forward/backward одного слоя;
//   * `forward`      — оркестратор параллельного forward;
//   * `backward`     — оркестратор параллельного backward.
//
// Наружу (в другие подсистемы `compute_manager`) экспортируются только
// оркестраторы и предикат `can_parallelize`; остальные символы доступны
// только внутри модуля `parallel`.

mod plan;
mod tracker;
mod shared;
mod chunk_ops;
mod dims;
mod dispatch;
mod forward;
mod backward;

pub(crate) use dispatch::can_parallelize;
pub(crate) use forward::forward_universal_parallel;
pub(crate) use backward::backward_universal_parallel;