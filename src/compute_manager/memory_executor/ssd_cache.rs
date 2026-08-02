// src/compute_manager/memory_executor/ssd_cache.rs

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use super::executor::MemoryError;

/// Дескриптор выделенного блока на SSD.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsdHandle {
    pub file_path: PathBuf,
    pub elements: usize,
}

/// Менеджер SSD-кэша.
pub struct SsdCacheManager {
    base_path: PathBuf,
    max_bytes: u64,
    used_bytes: AtomicU64,
    next_id: AtomicU64,
    _fs_lock: Mutex<()>,
}

impl SsdCacheManager {
    pub fn new(base_path: PathBuf, max_bytes: u64) -> Result<Self, MemoryError> {
        fs::create_dir_all(&base_path)
            .map_err(|e| MemoryError::SsdError(format!("Не удалось создать директорию SSD-кэша: {}", e)))?;

        let mut used: u64 = 0;
        if let Ok(entries) = fs::read_dir(&base_path) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() {
                        used += meta.len();
                    }
                }
            }
        }

        if used > max_bytes {
            return Err(MemoryError::SsdError(format!(
                "Текущий размер SSD-кэша ({} байт) превышает заданный лимит ({} байт)",
                used, max_bytes
            )));
        }

        Ok(Self {
            base_path,
            max_bytes,
            used_bytes: AtomicU64::new(used),
            next_id: AtomicU64::new(0),
            _fs_lock: Mutex::new(()),
        })
    }

    pub fn can_allocate(&self, elements: usize) -> bool {
        let required = elements as u64 * 4;
        let current = self.used_bytes.load(Ordering::Relaxed);
        current + required <= self.max_bytes
    }

    pub fn allocate(&self, elements: usize) -> Result<SsdHandle, MemoryError> {
        let required = elements as u64 * 4;
        let current = self.used_bytes.load(Ordering::Relaxed);
        if current + required > self.max_bytes {
            return Err(MemoryError::OutOfMemory(
                crate::compute_manager::memory_executor::types::MemoryDeviceKind::SsdCache,
            ));
        }

        let _lock = self._fs_lock.lock().unwrap();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let file_name = format!("buffer_{:08x}.dat", id);
        let file_path = self.base_path.join(&file_name);

        {
            let mut file = File::create(&file_path).map_err(|e| {
                MemoryError::SsdError(format!("Не удалось создать файл SSD-кэша: {}", e))
            })?;
            let zero_block = vec![0u8; required as usize];
            file.write_all(&zero_block).map_err(|e| {
                let _ = fs::remove_file(&file_path);
                MemoryError::SsdError(format!("Ошибка записи в файл SSD-кэша: {}", e))
            })?;
            file.flush().map_err(|e| {
                let _ = fs::remove_file(&file_path);
                MemoryError::SsdError(format!("Ошибка сброса файла SSD-кэша: {}", e))
            })?;
        }

        self.used_bytes.fetch_add(required, Ordering::Relaxed);

        Ok(SsdHandle {
            file_path,
            elements,
        })
    }

    pub fn deallocate(&self, handle: &SsdHandle) -> Result<(), MemoryError> {
        let _lock = self._fs_lock.lock().unwrap();
        if handle.file_path.exists() {
            let size = fs::metadata(&handle.file_path)
                .map(|m| m.len())
                .unwrap_or(0);
            fs::remove_file(&handle.file_path).map_err(|e| {
                MemoryError::SsdError(format!("Не удалось удалить файл SSD-кэша: {}", e))
            })?;
            self.used_bytes.fetch_sub(size, Ordering::Relaxed);
        }
        Ok(())
    }

    pub fn read(&self, handle: &SsdHandle) -> Result<Vec<f32>, MemoryError> {
        let _lock = self._fs_lock.lock().unwrap();
        let mut file = File::open(&handle.file_path).map_err(|e| {
            MemoryError::SsdError(format!("Не удалось открыть файл SSD-кэша для чтения: {}", e))
        })?;
        let expected_bytes = handle.elements * 4;
        let mut buf = vec![0u8; expected_bytes];
        file.read_exact(&mut buf).map_err(|e| {
            MemoryError::SsdError(format!("Ошибка чтения файла SSD-кэша: {}", e))
        })?;
        let floats: Vec<f32> = buf
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Ok(floats)
    }

    pub fn write(&self, handle: &SsdHandle, data: &[f32]) -> Result<(), MemoryError> {
        let _lock = self._fs_lock.lock().unwrap();
        if data.len() != handle.elements {
            return Err(MemoryError::SsdError(format!(
                "Несоответствие размера данных: ожидалось {} элементов, получено {}",
                handle.elements, data.len()
            )));
        }
        let byte_slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * 4,
            )
        };
        let mut file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&handle.file_path)
            .map_err(|e| MemoryError::SsdError(format!("Не удалось открыть файл SSD-кэша для записи: {}", e)))?;
        file.write_all(byte_slice).map_err(|e| {
            MemoryError::SsdError(format!("Ошибка записи в файл SSD-кэша: {}", e))
        })?;
        file.flush().map_err(|e| {
            MemoryError::SsdError(format!("Ошибка сброса файла SSD-кэша: {}", e))
        })?;
        Ok(())
    }
}