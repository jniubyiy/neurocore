// src/compute_manager/graph/mod.rs
//
// Подмодуль `graph` сохраняет только `types` — общий словарь контекстов
// слоёв (`DynamicContext`, `ChunkedContexts`), которым пользуется и v2
// (через `gpu::processor` и `graph_v2`).
//
// Модули v1 (`model`, `builder`, `forward`, `backward`) удалены на
// этапе D вместе с `MixedModel` и `ComputeExecutor`.

pub mod types;