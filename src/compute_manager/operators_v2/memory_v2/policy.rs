// src/compute_manager/memory_executor/policy.rs

use std::time::{Duration, Instant};

/// Уровень памяти (иерархия).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// Медленное, но ёмкое хранилище (SSD или диск).
    Ssd,
    /// Оперативная память (RAM).
    Ram,
    /// Видеопамять (VRAM) — самое быстрое, но ограниченное.
    Vram,
}

/// Приоритет буфера (влияет на решение о перемещении).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferPriority {
    /// Критичные данные (например, параметры модели) — стараемся держать в VRAM.
    High,
    /// Обычные данные (активации) — могут перемещаться.
    Medium,
    /// Низкоприоритетные (промежуточные, кэш) — выгружаются в первую очередь.
    Low,
}

/// Метаданные буфера для принятия решений о перемещении.
#[derive(Debug, Clone)]
pub struct BufferMetadata {
    /// Время последнего доступа.
    pub last_access: Instant,
    /// Количество обращений за последний период (сбрасывается каждый tick).
    pub access_count: usize,
    /// Общее количество обращений за всё время.
    pub total_access_count: usize,
    /// Размер в элементах f32 (или байтах, но будем считать элементами).
    pub size_elements: usize,
    /// Приоритет буфера.
    pub priority: BufferPriority,
}

impl BufferMetadata {
    pub fn new(size_elements: usize, priority: BufferPriority) -> Self {
        Self {
            last_access: Instant::now(),
            access_count: 0,
            total_access_count: 0,
            size_elements,
            priority,
        }
    }

    /// Обновить статистику при обращении к буферу.
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
        self.access_count += 1;
        self.total_access_count += 1;
    }

    /// Сбросить счётчик обращений за период (вызывается после каждого tick).
    pub fn reset_period_counter(&mut self) {
        self.access_count = 0;
    }
}

/// Политика автоматического управления памятью.
pub struct MemoryPolicy {
    /// Порог заполнения VRAM (0.0..1.0), выше которого начинаем выгружать.
    pub vram_high_watermark: f32,
    /// Порог, ниже которого можно загружать новые данные в VRAM.
    pub vram_low_watermark: f32,
    /// Время неиспользования (в секундах), после которого буфер может быть выгружен на SSD.
    pub ssd_eviction_age_secs: u64,
    /// Минимальное число обращений за период для продвижения в VRAM.
    pub promotion_threshold: usize,
    /// Максимальный размер буфера (в элементах), который мы готовы держать в VRAM
    /// (буферы больше этого размера могут быть перемещены в RAM/SSD быстрее).
    pub max_vram_buffer_elements: usize,
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        Self {
            vram_high_watermark: 0.8,
            vram_low_watermark: 0.4,
            ssd_eviction_age_secs: 60,
            promotion_threshold: 5,
            max_vram_buffer_elements: 10_000_000, // ~40 МБ (10 млн f32)
        }
    }
}

impl MemoryPolicy {
    /// Создать политику с настраиваемыми параметрами.
    pub fn new(
        vram_high_watermark: f32,
        vram_low_watermark: f32,
        ssd_eviction_age_secs: u64,
        promotion_threshold: usize,
        max_vram_buffer_elements: usize,
    ) -> Self {
        Self {
            vram_high_watermark,
            vram_low_watermark,
            ssd_eviction_age_secs,
            promotion_threshold,
            max_vram_buffer_elements,
        }
    }

    /// Принять решение о перемещении буфера на основе его метаданных и текущей загрузки памяти.
    ///
    /// # Аргументы
    /// * `metadata` — метаданные буфера.
    /// * `current_tier` — текущий уровень памяти, где находится буфер.
    /// * `vram_usage` — текущее использование VRAM (0.0..1.0).
    /// * `ram_usage` — текущее использование RAM (0.0..1.0).
    ///
    /// # Возвращает
    /// * `Some(MemoryTier)` — целевой уровень, на который следует переместить буфер.
    /// * `None` — перемещение не требуется.
    pub fn decide_movement(
        &self,
        metadata: &BufferMetadata,
        current_tier: MemoryTier,
        vram_usage: f32,
        ram_usage: f32,
    ) -> Option<MemoryTier> {
        // Приоритетная логика:
        // 1. Если буфер закреплён (pinned), мы не перемещаем его.
        //    Но здесь мы не знаем о pinned, это будет учитываться в executor.
        //    В policy мы просто принимаем решение на основе метаданных.

        // Если буфер очень маленький, его можно оставить в любом месте, но мы стремимся
        // хранить часто используемые маленькие буферы в VRAM.
        let is_small = metadata.size_elements < 1024; // < 4 КБ

        // Определяем, насколько буфер "горячий" (часто используемый)
        let is_hot = metadata.access_count >= self.promotion_threshold
            || (metadata.total_access_count > 100 && metadata.access_count > 0);

        let is_recent = metadata.last_access.elapsed() < Duration::from_secs(self.ssd_eviction_age_secs);

        match current_tier {
            MemoryTier::Vram => {
                // Буфер уже в VRAM. Решаем, не выгрузить ли его.
                // Выгружаем, если:
                // - VRAM переполнена (выше high_watermark)
                // - буфер не горячий и не очень недавно использовался
                // - или буфер большой и не очень горячий
                let should_evict = (vram_usage > self.vram_high_watermark)
                    && (!is_hot || !is_recent)
                    && (metadata.priority != BufferPriority::High);

                if should_evict {
                    // Если RAM достаточно свободна, выгружаем в RAM, иначе в SSD
                    if ram_usage < 0.7 {
                        Some(MemoryTier::Ram)
                    } else {
                        Some(MemoryTier::Ssd)
                    }
                } else {
                    None
                }
            }
            MemoryTier::Ram => {
                // Буфер в RAM. Решаем, продвинуть ли в VRAM или выгрузить на SSD.
                // Продвигаем в VRAM, если:
                // - VRAM не переполнена (ниже low_watermark)
                // - буфер горячий и приоритет высокий или средний
                // - буфер не слишком большой
                let can_promote = (vram_usage < self.vram_low_watermark)
                    && is_hot
                    && (metadata.priority != BufferPriority::Low)
                    && (metadata.size_elements <= self.max_vram_buffer_elements || is_small);

                if can_promote {
                    Some(MemoryTier::Vram)
                } else if !is_recent && (metadata.priority == BufferPriority::Low) {
                    // Если буфер старый и низкоприоритетный, можно выгрузить на SSD
                    Some(MemoryTier::Ssd)
                } else {
                    None
                }
            }
            MemoryTier::Ssd => {
                // Буфер на SSD. Решаем, продвинуть ли в RAM или VRAM.
                // Продвигаем, если:
                // - буфер горячий или недавно использовался
                // - есть место в RAM (или VRAM)
                if is_hot && ram_usage < 0.8 {
                    // Продвигаем в RAM
                    Some(MemoryTier::Ram)
                } else if is_hot && vram_usage < self.vram_low_watermark
                    && metadata.priority != BufferPriority::Low
                    && metadata.size_elements <= self.max_vram_buffer_elements
                {
                    // Сразу в VRAM, если есть место
                    Some(MemoryTier::Vram)
                } else {
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_metadata(size: usize, priority: BufferPriority, age_secs: u64) -> BufferMetadata {
        let mut meta = BufferMetadata::new(size, priority);
        // Имитируем возраст: если age_secs > 0, сдвигаем last_access назад
        if age_secs > 0 {
            meta.last_access = Instant::now() - Duration::from_secs(age_secs);
        }
        meta
    }

    #[test]
    fn test_vram_eviction() {
        let policy = MemoryPolicy::default();
        let meta = create_metadata(1024, BufferPriority::Medium, 10);
        // VRAM переполнена, буфер не горячий (access_count=0)
        let decision = policy.decide_movement(&meta, MemoryTier::Vram, 0.9, 0.5);
        assert!(decision.is_some());
        // Должен быть RAM или SSD
        let tier = decision.unwrap();
        assert!(tier == MemoryTier::Ram || tier == MemoryTier::Ssd);
    }

    #[test]
    fn test_promotion_to_vram() {
        let policy = MemoryPolicy::default();
        let mut meta = create_metadata(1024, BufferPriority::High, 0);
        // Имитируем много обращений
        meta.access_count = 10;
        let decision = policy.decide_movement(&meta, MemoryTier::Ram, 0.3, 0.5);
        assert_eq!(decision, Some(MemoryTier::Vram));
    }

    #[test]
    fn test_ssd_eviction() {
        let policy = MemoryPolicy::default();
        let meta = create_metadata(1024, BufferPriority::Low, 120); // старый, низкий приоритет
        let decision = policy.decide_movement(&meta, MemoryTier::Ram, 0.5, 0.9);
        assert_eq!(decision, Some(MemoryTier::Ssd));
    }

    #[test]
    fn test_no_move() {
        let policy = MemoryPolicy::default();
        let meta = create_metadata(1024, BufferPriority::High, 0);
        // VRAM занята не сильно, буфер горячий — оставляем
        let decision = policy.decide_movement(&meta, MemoryTier::Vram, 0.5, 0.5);
        assert_eq!(decision, None);
    }
}