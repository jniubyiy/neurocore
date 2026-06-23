// src/model_plan/param_store/param_store.rs

const SEGMENT_SIZE: usize = 1024; // размер одного сегмента

/// Дескриптор непрерывного участка в общем массиве параметров.
#[derive(Debug, Clone, Copy)]
pub struct ParamSlice {
    pub start: usize,
    pub len: usize,
}

impl ParamSlice {
    pub fn new(start: usize, len: usize) -> Self {
        Self { start, len }
    }
}

use std::sync::{Arc, Mutex};

/// Сегментированное хранилище параметров с отдельными Mutex на каждый сегмент.
#[derive(Clone)]
pub struct ParamStore {
    segments: Arc<Vec<Mutex<Vec<f32>>>>,
    total_len: usize,
}

impl ParamStore {
    pub fn new() -> Self {
        Self {
            segments: Arc::new(Vec::new()),
            total_len: 0,
        }
    }

    /// Выделяет непрерывный блок заданной длины (заполняется нулями) и возвращает его дескриптор.
    pub fn allocate(&mut self, len: usize) -> ParamSlice {
        let start = self.total_len;
        self.total_len += len;

        // Добавляем параметры в сегменты
        let mut remaining = len;
        let mut offset = start;
        while remaining > 0 {
            let seg_idx = offset / SEGMENT_SIZE;
            let pos_in_seg = offset % SEGMENT_SIZE;
            let capacity = SEGMENT_SIZE - pos_in_seg;
            let take = remaining.min(capacity);

            // Если нужный сегмент ещё не создан, создаём
            if seg_idx >= self.segments.len() {
                let need = seg_idx - self.segments.len() + 1;
                let segments = Arc::get_mut(&mut self.segments).unwrap();
                for _ in 0..need {
                    segments.push(Mutex::new(vec![0.0; SEGMENT_SIZE]));
                }
            }

            offset += take;
            remaining -= take;
        }
        ParamSlice::new(start, len)
    }

    /// Выделяет блок и сразу заполняет его переданными значениями.
    pub fn allocate_with(&mut self, values: &[f32]) -> ParamSlice {
        let slice = self.allocate(values.len());
        self.set_slice(slice, values);
        slice
    }

    /// Доступ ко всему массиву параметров только для чтения (собирает из сегментов).
    pub fn all_params(&self) -> Vec<f32> {
        let mut result = vec![0.0; self.total_len];
        for (seg_idx, seg_mutex) in self.segments.iter().enumerate() {
            let seg = seg_mutex.lock().unwrap();
            let start = seg_idx * SEGMENT_SIZE;
            let len = seg.len().min(self.total_len - start);
            result[start..start + len].copy_from_slice(&seg[..len]);
        }
        result
    }

    /// Собирает все параметры в один плоский вектор (аналог all_params, но для единообразия назван иначе).
    pub fn all_params_vec(&self) -> Vec<f32> {
        self.all_params()
    }

    /// Записывает значения из плоского вектора обратно в сегменты.
    /// Предполагается, что длина values равна общему количеству параметров.
    pub fn set_all_params(&mut self, values: &[f32]) {
        assert_eq!(values.len(), self.total_len);
        let segments = Arc::get_mut(&mut self.segments).expect("Cannot get mutable access");
        let mut offset = 0;
        for seg_idx in 0..segments.len() {
            let start = seg_idx * SEGMENT_SIZE;
            let len = segments[seg_idx].get_mut().unwrap().len().min(self.total_len - start);
            segments[seg_idx].get_mut().unwrap()[..len].copy_from_slice(&values[offset..offset + len]);
            offset += len;
        }
    }

    /// Применяет градиент ко всем параметрам: p -= lr * g (последовательно, для обратной совместимости)
    pub fn apply_gradient(&mut self, lr: f32, grad: &[f32]) {
        assert_eq!(self.total_len, grad.len(), "ParamStore: grad length mismatch");
        let segments = Arc::get_mut(&mut self.segments).expect("Cannot get mutable access");
        let mut grad_offset = 0;
        for seg_idx in 0..segments.len() {
            let start = seg_idx * SEGMENT_SIZE;
            let len = segments[seg_idx].get_mut().unwrap().len().min(self.total_len - start);
            let seg = segments[seg_idx].get_mut().unwrap();
            for (p, &g) in seg[..len].iter_mut().zip(&grad[grad_offset..grad_offset + len]) {
                *p -= lr * g;
            }
            grad_offset += len;
        }
    }

    /// Применяет градиент к диапазону параметров, блокируя только нужные сегменты.
    pub fn apply_gradient_slice(&self, lr: f32, grad: &[f32], start: usize) {
        let end = start + grad.len();
        assert!(end <= self.total_len);
        let mut pos = start;
        let mut grad_offset = 0;
        while pos < end {
            let seg_idx = pos / SEGMENT_SIZE;
            let seg_start = seg_idx * SEGMENT_SIZE;
            let seg_end = (seg_start + SEGMENT_SIZE).min(self.total_len);
            let in_seg_start = pos - seg_start;
            let in_seg_end = (end.min(seg_end)) - seg_start;
            let mut seg = self.segments[seg_idx].lock().unwrap();
            for (p, &g) in seg[in_seg_start..in_seg_end].iter_mut().zip(&grad[grad_offset..grad_offset + in_seg_end - in_seg_start]) {
                *p -= lr * g;
            }
            pos += in_seg_end - in_seg_start;
            grad_offset += in_seg_end - in_seg_start;
        }
    }

    /// Количество параметров в хранилище.
    pub fn len(&self) -> usize {
        self.total_len
    }

    /// Доступ к конкретному параметру по индексу.
    pub fn get(&self, index: usize) -> f32 {
        let seg_idx = index / SEGMENT_SIZE;
        let pos = index % SEGMENT_SIZE;
        self.segments[seg_idx].lock().unwrap()[pos]
    }

    /// Установить значение одного параметра по глобальному индексу.
    pub fn set_param(&mut self, index: usize, value: f32) {
        let seg_idx = index / SEGMENT_SIZE;
        let pos = index % SEGMENT_SIZE;
        let segments = Arc::get_mut(&mut self.segments).unwrap();
        segments[seg_idx].get_mut().unwrap()[pos] = value;
    }

    /// Заполнить слайс значениями из переданного слайса.
    pub fn set_slice(&mut self, slice: ParamSlice, values: &[f32]) {
        assert_eq!(values.len(), slice.len);
        let mut pos = slice.start;
        let mut val_offset = 0;
        while pos < slice.start + slice.len {
            let seg_idx = pos / SEGMENT_SIZE;
            let in_seg = pos % SEGMENT_SIZE;
            let seg_len = SEGMENT_SIZE - in_seg;
            let take = (slice.start + slice.len - pos).min(seg_len);
            let segments = Arc::get_mut(&mut self.segments).unwrap();
            let seg = segments[seg_idx].get_mut().unwrap();
            seg[in_seg..in_seg + take].copy_from_slice(&values[val_offset..val_offset + take]);
            pos += take;
            val_offset += take;
        }
    }

    /// Добавить delta к параметру с индексом `index`.
    pub fn add_to_param(&mut self, index: usize, delta: f32) {
        let seg_idx = index / SEGMENT_SIZE;
        let pos = index % SEGMENT_SIZE;
        let segments = Arc::get_mut(&mut self.segments).unwrap();
        segments[seg_idx].get_mut().unwrap()[pos] += delta;
    }
}