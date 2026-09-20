// src/layers/adapter/registry.rs

//! Реестр типов адаптеров — заглушка под будущий save/load.
//!
//! # Текущее состояние (Фаза 3 плана)
//!
//! Полноценная сериализация адаптеров (сохранение на диск, восстановление)
//! в проекте пока не реализована. Данный модуль — designated place для
//! неё: когда появится требование save/load (например, для персистентного
//! состояния адаптеров между сессиями), соответствующий код будет
//! добавлен сюда, а не размазан по слоям.
//!
//! # Зачем реестр нужен
//!
//! Адаптеры — гетерогенные (I-9). У каждого свой тип (структура),
//! своё состояние, свой алгоритм. При save/load нужно по имени типа
//! восстановить конкретную реализацию. Реестр обеспечивает единую
//! точку регистрации.
//!
//! До Фазы 3 модуль оставался заглушкой; API реализовано так, чтобы
//! его можно было расширить без ломающих изменений.

use std::collections::HashMap;
use std::sync::Mutex;

/// Метаданные типа адаптера.
#[derive(Debug, Clone, Default)]
pub struct AdapterTypeInfo {
    /// Человекочитаемое описание.
    pub description: String,

    /// Ожидаемый размер состояния на один параметр слоя (0 — stateless).
    pub state_size_per_param: usize,

    /// Версия формата состояния (для миграций в будущем).
    pub format_version: u32,
}

/// Глобальный реестр типов адаптеров.
///
/// Параллель `AdapterRegistry` в PyTorch-подобных системах, но
/// значительно проще: сейчас только хранит метаданные. Место для
/// будущей сериализации.
///
/// # Потокобезопасность
///
/// Внутреннее состояние — под `Mutex`. Все методы берут `&self`.
/// Реестр обычно инстанцируется один раз на сессию.
pub struct AdapterRegistry {
    entries: Mutex<HashMap<&'static str, AdapterTypeInfo>>,
}

impl AdapterRegistry {
    /// Создаёт пустой реестр.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Регистрирует тип адаптера под данным именем.
    ///
    /// Если имя уже занято — перезаписывает (это удобно при hot-reload
    /// или тестировании).
    pub fn register(&self, name: &'static str, info: AdapterTypeInfo) {
        self.entries.lock().unwrap().insert(name, info);
    }

    /// Возвращает `true`, если тип с таким именем зарегистрирован.
    pub fn is_registered(&self, name: &str) -> bool {
        self.entries.lock().unwrap().contains_key(name)
    }

    /// Возвращает метаданные типа, если он зарегистрирован.
    pub fn get(&self, name: &str) -> Option<AdapterTypeInfo> {
        self.entries.lock().unwrap().get(name).cloned()
    }

    /// Количество зарегистрированных типов.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    /// `true`, если реестр пуст.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Юнит-тесты
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry_is_inert() {
        let reg = AdapterRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(!reg.is_registered("learnable_mish"));
        assert!(reg.get("learnable_mish").is_none());
    }

    #[test]
    fn register_and_lookup() {
        let reg = AdapterRegistry::new();
        reg.register(
            "learnable_mish",
            AdapterTypeInfo {
                description: "Mish gradient adapter".to_string(),
                state_size_per_param: 0,
                format_version: 1,
            },
        );
        assert!(reg.is_registered("learnable_mish"));
        assert_eq!(reg.len(), 1);
        let info = reg.get("learnable_mish").unwrap();
        assert_eq!(info.description, "Mish gradient adapter");
        assert_eq!(info.format_version, 1);
    }
}