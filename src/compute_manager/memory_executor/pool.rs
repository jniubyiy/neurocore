// src/compute_manager/memory_executor/pool.rs

/// Пул памяти одного устройства (CPU, GPU или SSD). Учитывает занятое количество элементов f32.
#[derive(Debug)]
pub struct MemoryPool {
    /// Максимальное количество элементов f32, которое разрешено занять
    pub max_elements: usize,
    /// Текущее количество занятых элементов
    pub used_elements: usize,
}

impl MemoryPool {
    pub fn new(max_memory_bytes: u64) -> Self {
        // Переводим байты в количество f32 (4 байта на элемент)
        let max_elements = (max_memory_bytes / 4) as usize;
        MemoryPool {
            max_elements,
            used_elements: 0,
        }
    }

    /// Проверить, можно ли выделить указанное количество элементов
    pub fn can_allocate(&self, elements: usize) -> bool {
        self.used_elements + elements <= self.max_elements
    }

    /// Зарезервировать память (увеличивает счётчик)
    pub fn allocate(&mut self, elements: usize) -> Result<(), String> {
        if self.can_allocate(elements) {
            self.used_elements += elements;
            Ok(())
        } else {
            Err(format!(
                "Memory pool exhausted: needed {}, available {}",
                elements,
                self.max_elements - self.used_elements
            ))
        }
    }

    /// Освободить память (уменьшает счётчик)
    pub fn deallocate(&mut self, elements: usize) {
        self.used_elements = self.used_elements.saturating_sub(elements);
    }
}