// src/neuron/types/splitter.rs

/// Нейрон-разделитель: принимает один вектор `x` длины `n` и выдаёт два вектора `a` (длины `p`) и `b` (длины `q`).
/// Параметры: матрицы весов `W_a` (p×n), `W_b` (q×n) и смещения `bias_a` (p), `bias_b` (q).
/// Формула: a = activation(W_a * x + bias_a), b = activation(W_b * x + bias_b). По умолчанию ReLU.
pub struct Splitter {
    pub n: usize,         // размерность входного вектора
    pub p: usize,         // размерность первого выходного вектора
    pub q: usize,         // размерность второго выходного вектора
    pub wa: Vec<f32>,     // веса для a, размер p * n
    pub wb: Vec<f32>,     // веса для b, размер q * n
    pub bias_a: Vec<f32>, // смещения для a, длина p
    pub bias_b: Vec<f32>, // смещения для b, длина q
}

impl Splitter {
    /// Создаёт разделитель с заданными размерностями, параметры инициализируются нулями.
    pub fn new(n: usize, p: usize, q: usize) -> Self {
        Self {
            n,
            p,
            q,
            wa: vec![0.0; p * n],
            wb: vec![0.0; q * n],
            bias_a: vec![0.0; p],
            bias_b: vec![0.0; q],
        }
    }

    /// Общее количество параметров.
    pub fn param_count(&self) -> usize {
        self.p * self.n + self.q * self.n + self.p + self.q
    }

    /// Сериализация параметров в плоский вектор.
    pub fn get_params(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.param_count());
        v.extend_from_slice(&self.wa);
        v.extend_from_slice(&self.wb);
        v.extend_from_slice(&self.bias_a);
        v.extend_from_slice(&self.bias_b);
        v
    }

    /// Установка параметров из плоского вектора.
    pub fn set_params(&mut self, values: &[f32]) {
        let wa_len = self.p * self.n;
        let wb_len = self.q * self.n;
        let ba_len = self.p;
        let bb_len = self.q;
        assert_eq!(values.len(), wa_len + wb_len + ba_len + bb_len);
        self.wa.copy_from_slice(&values[..wa_len]);
        self.wb.copy_from_slice(&values[wa_len..wa_len + wb_len]);
        self.bias_a.copy_from_slice(&values[wa_len + wb_len..wa_len + wb_len + ba_len]);
        self.bias_b.copy_from_slice(&values[wa_len + wb_len + ba_len..]);
    }

    /// Прямой проход для одного образца: выдаёт (a, b) и кэш (x, pre_a, pre_b).
    pub fn forward_sample(&self, x: &[f32], a_out: &mut [f32], b_out: &mut [f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        assert_eq!(x.len(), self.n);
        assert_eq!(a_out.len(), self.p);
        assert_eq!(b_out.len(), self.q);
        let mut pre_a = vec![0.0; self.p];
        let mut pre_b = vec![0.0; self.q];
        // Вычисляем a
        for i in 0..self.p {
            let mut sum = self.bias_a[i];
            for j in 0..self.n {
                sum += self.wa[i * self.n + j] * x[j];
            }
            pre_a[i] = sum;
            a_out[i] = sum.max(0.0); // ReLU
        }
        // Вычисляем b
        for i in 0..self.q {
            let mut sum = self.bias_b[i];
            for j in 0..self.n {
                sum += self.wb[i * self.n + j] * x[j];
            }
            pre_b[i] = sum;
            b_out[i] = sum.max(0.0); // ReLU
        }
        (x.to_vec(), pre_a, pre_b)
    }

    /// Обратный проход для одного образца: получает градиенты по выходам `da`, `db`, кэш,
    /// возвращает градиент по входу `dx` и градиенты параметров (плоский вектор).
    pub fn backward_sample(&self, da: &[f32], db: &[f32], cache: &(Vec<f32>, Vec<f32>, Vec<f32>)) -> (Vec<f32>, Vec<f32>) {
        let (x, pre_a, pre_b) = cache;
        let mut dx = vec![0.0; self.n];
        let mut d_wa = vec![0.0; self.p * self.n];
        let mut d_wb = vec![0.0; self.q * self.n];
        let mut d_bias_a = vec![0.0; self.p];
        let mut d_bias_b = vec![0.0; self.q];

        // Градиенты от a
        for i in 0..self.p {
            let d_pre = if pre_a[i] > 0.0 { da[i] } else { 0.0 };
            d_bias_a[i] += d_pre;
            for j in 0..self.n {
                dx[j] += d_pre * self.wa[i * self.n + j];
                d_wa[i * self.n + j] = d_pre * x[j];
            }
        }
        // Градиенты от b
        for i in 0..self.q {
            let d_pre = if pre_b[i] > 0.0 { db[i] } else { 0.0 };
            d_bias_b[i] += d_pre;
            for j in 0..self.n {
                dx[j] += d_pre * self.wb[i * self.n + j];
                d_wb[i * self.n + j] = d_pre * x[j];
            }
        }

        let mut d_params = Vec::with_capacity(self.param_count());
        d_params.extend_from_slice(&d_wa);
        d_params.extend_from_slice(&d_wb);
        d_params.extend_from_slice(&d_bias_a);
        d_params.extend_from_slice(&d_bias_b);
        (dx, d_params)
    }

    /// Пакетный прямой проход: на вход матрица X (batch × n), на выходе два набора матриц A и B.
    pub fn forward_batch(&self, x_batch: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>) {
        let batch = x_batch.len();
        let mut a_batch = Vec::with_capacity(batch);
        let mut b_batch = Vec::with_capacity(batch);
        let mut caches = Vec::with_capacity(batch);
        for i in 0..batch {
            let mut a = vec![0.0; self.p];
            let mut b = vec![0.0; self.q];
            let cache = self.forward_sample(&x_batch[i], &mut a, &mut b);
            a_batch.push(a);
            b_batch.push(b);
            caches.push(cache);
        }
        (a_batch, b_batch, caches)
    }

    /// Пакетный обратный проход.
    pub fn backward_batch(
        &self,
        da_batch: &[Vec<f32>],
        db_batch: &[Vec<f32>],
        caches: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        let batch = da_batch.len();
        let mut dx_batch = Vec::with_capacity(batch);
        let mut d_params = vec![0.0; self.param_count()];
        for i in 0..batch {
            let (dx, dpar) = self.backward_sample(&da_batch[i], &db_batch[i], &caches[i]);
            dx_batch.push(dx);
            for (idx, &val) in dpar.iter().enumerate() {
                d_params[idx] += val;
            }
        }
        (dx_batch, d_params)
    }
}