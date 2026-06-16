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

/// Централизованное хранилище всех параметров модели.
#[derive(Debug, Clone)]
pub struct ParamStore {
    data: Vec<f32>,
}

impl ParamStore {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Выделяет непрерывный блок заданной длины (заполняется нулями) и возвращает его дескриптор.
    pub fn allocate(&mut self, len: usize) -> ParamSlice {
        let start = self.data.len();
        self.data.resize(start + len, 0.0);
        ParamSlice::new(start, len)
    }

    /// Выделяет блок и сразу заполняет его переданными значениями.
    pub fn allocate_with(&mut self, values: &[f32]) -> ParamSlice {
        let slice = self.allocate(values.len());
        self.data[slice.start..slice.start + slice.len].copy_from_slice(values);
        slice
    }

    /// Доступ ко всему массиву параметров только для чтения.
    pub fn all_params(&self) -> &[f32] {
        &self.data
    }

    /// Применяет градиент ко всем параметрам: p -= lr * g
    pub fn apply_gradient(&mut self, lr: f32, grad: &[f32]) {
        assert_eq!(self.data.len(), grad.len(), "ParamStore: grad length mismatch");
        for (p, g) in self.data.iter_mut().zip(grad) {
            *p -= lr * g;
        }
    }

    /// Количество параметров в хранилище.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Доступ к конкретному параметру по индексу.
    pub fn get(&self, index: usize) -> f32 {
        self.data[index]
    }

    /// Установить значение одного параметра по глобальному индексу.
    pub fn set_param(&mut self, index: usize, value: f32) {
        self.data[index] = value;
    }

    /// Заполнить слайс значениями из переданного слайса.
    pub fn set_slice(&mut self, slice: ParamSlice, values: &[f32]) {
        assert_eq!(values.len(), slice.len);
        self.data[slice.start..slice.start + slice.len].copy_from_slice(values);
    }
}