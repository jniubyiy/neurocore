// src/tensor/tensor2d.rs

#[derive(Debug, Clone)]
pub struct Tensor2D {
    pub dim1: usize,
    pub dim2: usize,
    pub data: Vec<Vec<f32>>,

    /// Длины реальных данных в каждой строке.
    ///
    /// * `None` — тензор «плотный»: все строки имеют длину `dim2`.
    /// * `Some(lens)` — тензор «ragged»: `lens[i]` — длина реальных
    ///   данных в строке `i`, при этом физически строка хранится
    ///   длины `dim2`, а «хвост» `lens[i]..dim2` заполнен нулями
    ///   (padding). Потребители, которые не знают про ragged, видят
    ///   просто нули и работают как обычно. Слой, знающий про ragged
    ///   (`AdaptiveSpaceCompress`), читает `sample_lens` и обрабатывает
    ///   только реальную часть.
    ///
    /// Инвариант: `sample_lens.len() == dim1`, и `lens[i] <= dim2`.
    pub sample_lens: Option<Vec<usize>>,
}

impl Tensor2D {
    /// Создаёт плотный тензор. Паникует, если длины строк разные.
    pub fn new(data: Vec<Vec<f32>>) -> Self {
        let dim1 = data.len();
        let dim2 = if dim1 > 0 { data[0].len() } else { 0 };
        for (i, row) in data.iter().enumerate() {
            if row.len() != dim2 {
                panic!(
                    "Tensor2D::new: row {} has length {}, expected {} (dim1 = {})",
                    i,
                    row.len(),
                    dim2,
                    dim1
                );
            }
        }
        Tensor2D {
            dim1,
            dim2,
            data,
            sample_lens: None,
        }
    }

    /// Создаёт ragged-тензор из примеров произвольной длины.
    ///
    /// Все строки паддятся нулями до `dim2 = max(sample_lens)`.
    /// `sample_lens[i]` хранит **исходную** (не padding) длину примера `i`.
    pub fn ragged(samples: Vec<Vec<f32>>) -> Self {
        let dim1 = samples.len();
        let dim2 = samples.iter().map(|s| s.len()).max().unwrap_or(0);
        let sample_lens: Vec<usize> = samples.iter().map(|s| s.len()).collect();
        let data: Vec<Vec<f32>> = samples
            .into_iter()
            .map(|mut s| {
                s.resize(dim2, 0.0);
                s
            })
            .collect();
        Tensor2D {
            dim1,
            dim2,
            data,
            sample_lens: Some(sample_lens),
        }
    }

    pub fn zeros(dim1: usize, dim2: usize) -> Self {
        Tensor2D {
            dim1,
            dim2,
            data: vec![vec![0.0; dim2]; dim1],
            sample_lens: None,
        }
    }

    pub fn row(&self, r: usize) -> Vec<f32> {
        self.data[r].clone()
    }

    pub fn from_scalar(value: f32) -> Self {
        Tensor2D {
            dim1: 1,
            dim2: 1,
            data: vec![vec![value]],
            sample_lens: None,
        }
    }

    /// `true`, если тензор хранит примеры произвольной длины.
    #[inline]
    pub fn is_ragged(&self) -> bool {
        self.sample_lens.is_some()
    }

    /// Возвращает длины реальных данных каждого примера.
    /// `None` для плотного тензора.
    #[inline]
    pub fn sample_lens(&self) -> Option<&[usize]> {
        self.sample_lens.as_deref()
    }

    /// Возвращает срез тензора по строкам `[start, end)`.
    ///
    /// Сохраняет `sample_lens` (в нарезке), если тензор ragged.
    pub fn slice_rows(&self, start: usize, end: usize) -> Self {
        assert!(end <= self.dim1, "slice_rows: end > dim1");
        let data = self.data[start..end].to_vec();
        let sample_lens = self
            .sample_lens
            .as_ref()
            .map(|lens| lens[start..end].to_vec());
        Tensor2D {
            dim1: end - start,
            dim2: self.dim2,
            data,
            sample_lens,
        }
    }
}




