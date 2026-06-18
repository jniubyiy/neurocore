// src/neuron/types/combiner.rs

/// Нейрон-объединитель: принимает два вектора `a` и `b` (одинаковой длины `n`) и выдаёт вектор `c` длины `m`.
/// Параметры: матрицы весов `W_a` (m×n), `W_b` (m×n) и смещения `bias` (m).
/// Формула: c = activation(W_a * a + W_b * b + bias), где activation – ReLU (можно заменить на Identity при необходимости).
pub struct Combiner {
    pub n: usize,         // размерность каждого из входных векторов
    pub m: usize,         // размерность выходного вектора
    pub wa: Vec<f32>,     // матрица весов для a, размер m * n, row-major: wa[i * n + j] – вес от a[j] к c[i]
    pub wb: Vec<f32>,     // матрица весов для b
    pub bias: Vec<f32>,   // смещения, длина m
}

impl Combiner {
    /// Создаёт объединитель с заданными размерностями. Параметры инициализируются нулями (или случайными малыми).
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            n,
            m,
            wa: vec![0.0; m * n],
            wb: vec![0.0; m * n],
            bias: vec![0.0; m],
        }
    }

    /// Количество обучаемых параметров.
    pub fn param_count(&self) -> usize {
        2 * self.m * self.n + self.m
    }

    /// Возвращает все параметры в виде линейного массива.
    pub fn get_params(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.param_count());
        v.extend_from_slice(&self.wa);
        v.extend_from_slice(&self.wb);
        v.extend_from_slice(&self.bias);
        v
    }

    /// Загружает параметры из линейного массива.
    pub fn set_params(&mut self, values: &[f32]) {
        let wa_len = self.m * self.n;
        let wb_len = wa_len;
        let bias_len = self.m;
        assert_eq!(values.len(), wa_len + wb_len + bias_len);
        self.wa.copy_from_slice(&values[..wa_len]);
        self.wb.copy_from_slice(&values[wa_len..wa_len + wb_len]);
        self.bias.copy_from_slice(&values[wa_len + wb_len..]);
    }

    /// Прямой проход для одного образца: принимает ссылки на векторы a и b, записывает результат в out (должен быть длины m).
    /// Возвращает (кэшированные данные для backward: сами векторы a, b и выход до активации).
    pub fn forward_sample(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        assert_eq!(a.len(), self.n);
        assert_eq!(b.len(), self.n);
        assert_eq!(out.len(), self.m);
        let mut pre_act = vec![0.0; self.m];
        for i in 0..self.m {
            let mut sum = self.bias[i];
            for j in 0..self.n {
                sum += self.wa[i * self.n + j] * a[j];
                sum += self.wb[i * self.n + j] * b[j];
            }
            pre_act[i] = sum;
            out[i] = sum.max(0.0); // ReLU активация
        }
        (a.to_vec(), b.to_vec(), pre_act)
    }

    /// Обратный проход для одного образца: получает градиент по выходу `d_out`, кэш из forward, и возвращает
    /// градиенты по входам `da`, `db` и градиенты по параметрам в виде плоского вектора.
    pub fn backward_sample(&self, d_out: &[f32], cache: &(Vec<f32>, Vec<f32>, Vec<f32>)) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (a, b, pre_act) = cache;
        let mut da = vec![0.0; self.n];
        let mut db = vec![0.0; self.n];
        let mut d_wa = vec![0.0; self.m * self.n];
        let mut d_wb = vec![0.0; self.m * self.n];
        let mut d_bias = vec![0.0; self.m];
        for i in 0..self.m {
            let d_pre = if pre_act[i] > 0.0 { d_out[i] } else { 0.0 }; // производная ReLU
            d_bias[i] += d_pre;
            for j in 0..self.n {
                da[j] += d_pre * self.wa[i * self.n + j];
                db[j] += d_pre * self.wb[i * self.n + j];
                d_wa[i * self.n + j] = d_pre * a[j];
                d_wb[i * self.n + j] = d_pre * b[j];
            }
        }
        let mut d_params = Vec::with_capacity(self.param_count());
        d_params.extend_from_slice(&d_wa);
        d_params.extend_from_slice(&d_wb);
        d_params.extend_from_slice(&d_bias);
        (da, db, d_params)
    }

    /// Пакетный прямой проход: принимает матрицы `A` (batch × n) и `B` (batch × n), возвращает матрицу выхода (batch × m).
    /// Также возвращает кэш для каждой строки (список кортежей).
    pub fn forward_batch(&self, a_batch: &[Vec<f32>], b_batch: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>) {
        let batch = a_batch.len();
        assert_eq!(b_batch.len(), batch);
        let mut outputs = Vec::with_capacity(batch);
        let mut caches = Vec::with_capacity(batch);
        for i in 0..batch {
            let mut out = vec![0.0; self.m];
            let cache = self.forward_sample(&a_batch[i], &b_batch[i], &mut out);
            outputs.push(out);
            caches.push(cache);
        }
        (outputs, caches)
    }

    /// Пакетный обратный проход: принимает градиент по выходу `d_out_batch` (batch × m), кэш из forward,
    /// возвращает градиенты по входам `dA` и `dB` и усреднённые градиенты параметров (суммированные по батчу).
    pub fn backward_batch(
        &self,
        d_out_batch: &[Vec<f32>],
        caches: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
        let batch = d_out_batch.len();
        let mut da_batch = Vec::with_capacity(batch);
        let mut db_batch = Vec::with_capacity(batch);
        let mut d_params = vec![0.0; self.param_count()];
        for i in 0..batch {
            let (da, db, dpar) = self.backward_sample(&d_out_batch[i], &caches[i]);
            da_batch.push(da);
            db_batch.push(db);
            for (idx, &val) in dpar.iter().enumerate() {
                d_params[idx] += val;
            }
        }
        (da_batch, db_batch, d_params)
    }
}