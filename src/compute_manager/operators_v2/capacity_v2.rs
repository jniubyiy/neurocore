// src/compute_manager/operators_v2/capacity_v2.rs
//
// Снимок загрузки оператора.
//
// Распределитель спрашивает `capacity()` у каждого оператора перед
// выбором, куда отправить job. Это единственная «обратная связь»,
// которую оператор даёт наружу — и она полностью неинвазивна:
// распределитель не заглядывает в потроха оператора, а лишь читает
// агрегированные числа.
//
// Семантика полей:
//   * `queue_len` — сколько job'ов сейчас в очереди (включая исполняемые);
//   * `max_queue_len` — лимит очереди, если он есть (None — без лимита);
//   * `is_ready` — принимает ли оператор новые job'ы (например, GPU-тред
//     жив и очередь не закрыта);
//   * `estimated_latency_ns` — оценка времени до возврата результата для
//     нового job'а. Для CPU — на основе mini-model scheduler'а (средняя
//     по недавним замерам); для GPU — средняя по последним замерам;
//     для Memory — None (обычно < 1 мс и не является узким местом).

use crate::compute_manager::jobs_v2::OperatorKind;

/// Снимок загрузки одного оператора.
#[derive(Debug, Clone, Copy)]
pub struct OperatorCapacity {
    /// Идентификатор оператора, выдавшего снимок.
    pub kind: OperatorKind,

    /// Сколько job'ов сейчас в очереди оператора.
    pub queue_len: usize,

    /// Лимит очереди (`None` — без явного ограничения).
    pub max_queue_len: Option<usize>,

    /// Готов ли оператор принимать новые job'ы.
    /// `false` — например, внутренний поток оператора упал или очередь
    /// закрыта.
    pub is_ready: bool,

    /// Оценка времени до возврата результата нового job'а (в наносекундах).
    /// `None` — оценка недоступна.
    pub estimated_latency_ns: Option<u64>,
}

impl OperatorCapacity {
    /// Оператор свободен и может принять job.
    #[inline]
    pub fn is_available(&self) -> bool {
        self.is_ready && !self.is_full()
    }

    /// Очередь оператора достигла лимита.
    #[inline]
    pub fn is_full(&self) -> bool {
        match self.max_queue_len {
            Some(max) => self.queue_len >= max,
            None => false,
        }
    }

    /// Доля использования очереди: `queue_len / max_queue_len`.
    /// Если лимит не задан, возвращает `0.0` (нечего оценивать).
    #[inline]
    pub fn utilization(&self) -> f32 {
        match self.max_queue_len {
            Some(0) => 0.0,
            Some(max) => (self.queue_len as f32 / max as f32).clamp(0.0, 1.0),
            None => 0.0,
        }
    }
}

impl OperatorCapacity {
    /// Снимок «оператор свободен, очередь пуста, латентность неизвестна».
    ///
    /// Удобный конструктор для операторов без собственной очереди
    /// (например, `MemoryOperatorV2`, у которого submit = исполнение).
    #[inline]
    pub fn idle(kind: OperatorKind) -> Self {
        Self {
            kind,
            queue_len: 0,
            max_queue_len: None,
            is_ready: true,
            estimated_latency_ns: None,
        }
    }

    /// Снимок «оператор недоступен».
    ///
    /// Используется, когда внутренний поток оператора упал или очередь
    /// закрыта: распределитель должен выбрать другого кандидата.
    #[inline]
    pub fn unavailable(kind: OperatorKind) -> Self {
        Self {
            kind,
            queue_len: 0,
            max_queue_len: None,
            is_ready: false,
            estimated_latency_ns: None,
        }
    }
}