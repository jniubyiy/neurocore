// src/layers/linear_attention/gpu/mod.rs

pub mod pipeline;

use std::collections::HashMap;
use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::compute_manager::gpu::compute::GpuCompute;
use crate::compute_manager::matrix_buffer::view::MatrixBufferView;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use vulkano::buffer::Subbuffer;

// ============================================================================
// Константы «резинки с горкой» (согласованы с CPU-версией)
// ============================================================================

/// Температура сигмоид в h_soft(h_raw). Меньше = резче переходы.
const H_SOFT_TAU: f32 = 0.1;

/// Коэффициент сглаживания w_h(h_soft). Меньше = резче граница головы.
const HEAD_WEIGHT_K: f32 = 8.0;

/// Порог, ниже которого head считается неактивным и не обрабатывается.
const HEAD_WEIGHT_EPS: f32 = 1e-4;

// ============================================================================
// Вспомогательные функции
// ============================================================================

/// Получает `Subbuffer<[f32]>` из `MatrixBufferView`, используя смещение
/// и длину. Родительский буфер должен быть GPU.
fn subbuffer_from_view(gpu: &GpuCompute, view: &MatrixBufferView) -> Subbuffer<[f32]> {
    let parent_sub = gpu.get_gpu_subbuffer_from_handle(view.parent_handle());
    let start = view.offset_elements() as u64;
    let end = (view.offset_elements() + view.len()) as u64;
    parent_sub.slice(start..end)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Мягкое число голов как функция от обучаемого h_raw.
///
///   h_soft(h_raw) = min_heads + Σ_{k=1}^{max−min} σ((h_raw − θ_k) / τ)
///   θ_k = k − 0.5
#[inline]
fn compute_h_soft(h_raw: f32, min_heads: usize, max_heads: usize) -> f32 {
    if min_heads >= max_heads {
        return min_heads as f32;
    }
    let n_trans = max_heads - min_heads;
    let mut h = min_heads as f32;
    for k in 1..=n_trans {
        let theta_k = (k as f32) - 0.5;
        h += sigmoid((h_raw - theta_k) / H_SOFT_TAU);
    }
    h
}

/// Производная h_soft по h_raw.
#[inline]
fn compute_h_soft_derivative(h_raw: f32, min_heads: usize, max_heads: usize) -> f32 {
    if min_heads >= max_heads {
        return 0.0;
    }
    let n_trans = max_heads - min_heads;
    let mut d = 0.0f32;
    for k in 1..=n_trans {
        let theta_k = (k as f32) - 0.5;
        let s = sigmoid((h_raw - theta_k) / H_SOFT_TAU);
        d += s * (1.0 - s) / H_SOFT_TAU;
    }
    d
}

/// Вес головы h при данном h_soft.
///
///   w_h = 0.5 · (1 + tanh(K · (h_soft − h − 0.5)))
#[inline]
fn head_weight(h_soft: f32, h: usize) -> f32 {
    let x = h_soft - h as f32;
    0.5 * (1.0 + ((x - 0.5) * HEAD_WEIGHT_K).tanh())
}

/// Производная w_h по h_soft.
#[inline]
fn head_weight_derivative(h_soft: f32, h: usize) -> f32 {
    let x = h_soft - h as f32;
    let t = ((x - 0.5) * HEAD_WEIGHT_K).tanh();
    0.5 * HEAD_WEIGHT_K * (1.0 - t * t)
}

/// Разбирает `params.len()` в `(max_heads, d_head)`.
///
/// Формат параметров LinearAttention в общем буфере:
///
///   для каждого head h = 0..max_heads−1 (порядок в буфере):
///     [Wq_h (dh·d_model) | bq_h (dh) | Wk_h | bk_h | Wv_h | bv_h |
///      Wo_h (d_model·dh) | bo_h (d_model) | self_bias_h (1)]
///   всего на head: head_param_count = 4·dh·d_model + 3·dh + d_model + 1
///   в конце — один скаляр h_raw.
///
/// Итого: max_heads · head_param_count + 1.
///
/// Так как dh = d_model / max_heads, а max_heads — делитель d_model,
/// перебираем всех делителей и ищем совпадение по длине.
fn parse_heads_from_params(params_len: usize, d_model: usize) -> (usize, usize) {
    assert!(d_model > 0, "LinearAttention GPU: d_model must be positive");
    for max_heads in 1..=d_model {
        if d_model % max_heads != 0 {
            continue;
        }
        let dh = d_model / max_heads;
        let hpc = 4 * dh * d_model + 3 * dh + d_model + 1;
        if params_len == max_heads * hpc + 1 {
            return (max_heads, dh);
        }
    }
    panic!(
        "LinearAttention GPU: params length {} not compatible with d_model={}",
        params_len, d_model
    );
}

// ============================================================================
// Кэш между forward и backward
// ============================================================================

/// Per-head буферы, сохраняемые между forward и backward.
struct GpuHeadCache {
    /// denom (batch, seq), column-major.
    denom: MatrixBufferHandle,
    /// attn (batch, seq·dh), column-major.
    attn: MatrixBufferHandle,
    /// y_h (batch, seq·d_model), column-major — выход головы до взвешивания.
    y_h: MatrixBufferHandle,
    /// Вес головы w_h = head_weight(h_soft, h) на момент forward.
    weight: f32,
    /// Индекс головы в общем списке (для сопоставления с параметрами).
    head_index: usize,
}

/// Полный кэш после forward одной LinearAttention.
struct ForwardCache {
    heads: Vec<GpuHeadCache>,
    h_raw: f32,
    h_soft: f32,
    batch: usize,
    seq_len: usize,
    d_model: usize,
}

/// Ключ кэша: (offset в родительском буфере параметров, id входного буфера).
///
/// `params.offset_elements()` уникален для каждого слоя внутри сегмента.
/// `input.id()` уникален для каждого forward-вызова (в пределах одной
/// тренировочной итерации). Вместе — однозначная идентификация forward-кэша.
type CacheKey = (usize, usize);

static FORWARD_CACHE: Lazy<Mutex<HashMap<CacheKey, ForwardCache>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[inline]
fn cache_key(params: &MatrixBufferView, input: &MatrixBufferHandle) -> CacheKey {
    (params.offset_elements(), input.id().0)
}

// ============================================================================
// Реализация методов GpuCompute
// ============================================================================

impl GpuCompute {
    // ------------------------------------------------------------------------
    // Вспомогательный диспетчер: per-token линейное преобразование.
    // ------------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn dispatch_linear_per_token(
        &self,
        in_buf: Subbuffer<[f32]>,
        weight_view: &MatrixBufferView,
        bias_view: &MatrixBufferView,
        out_buf: Subbuffer<[f32]>,
        batch: usize,
        seq_len: usize,
        in_features: usize,
        out_features: usize,
    ) {
        let w_buf = subbuffer_from_view(self, weight_view);
        let b_buf = subbuffer_from_view(self, bias_view);
        let pipeline = &self.linear_attention_pipelines().linear_per_token;
        let push = [
            batch as u32,
            seq_len as u32,
            in_features as u32,
            out_features as u32,
        ];
        let total = batch * seq_len * out_features;
        self.run_compute_shader(
            pipeline,
            &[(0, in_buf), (1, w_buf), (2, b_buf), (3, out_buf)],
            &push,
            total,
        );
    }

    // ------------------------------------------------------------------------
    // Multi-head forward.
    // ------------------------------------------------------------------------

    /// Прямой проход многоголового LinearAttention на GPU.
    ///
    /// Раскладка caller-allocated буферов `q_raw, k_raw, v_raw, q_phi, k_phi`
    /// (каждый размера `batch · seq_len · d_model` элементов):
    ///
    ///   [head 0: batch · seq_len · dh][head 1: ...][...]
    ///
    /// где `dh = d_model / max_heads`. Внутри одного head данные
    /// интерпретируются как `(batch, seq_len · dh)` column-major,
    /// ровно так, как ожидают шейдеры.
    ///
    /// Буферы `kv` (размер `d_model · d_model`) и `z` (размер `d_model`)
    /// интерпретируются как конкатенация per-head секций:
    ///   kv: [head 0: dh·dh][head 1: dh·dh]...   всего: max_heads · dh·dh ≤ d_model²
    ///   z:  [head 0: dh][head 1: dh]...         всего: max_heads · dh = d_model
    ///
    /// Per-head буферы `denom`, `attn`, `y_h` выделяются внутри метода и
    /// сохраняются в глобальном кэше для последующего backward.
    pub fn run_linear_attention_forward_buffered_handle_with_dims(
        &self,
        input: &MatrixBufferHandle,
        params: &MatrixBufferView,
        output: &MatrixBufferHandle,
        seq_len: usize,
        d_model: usize,
        q_raw: &MatrixBufferHandle,
        k_raw: &MatrixBufferHandle,
        v_raw: &MatrixBufferHandle,
        q_phi: &MatrixBufferHandle,
        k_phi: &MatrixBufferHandle,
        kv: &MatrixBufferHandle,
        z: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(output.is_gpu(), "Output handle must be GPU");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        assert_eq!(input.cols(), seq_len * d_model, "Input cols mismatch");
        assert_eq!(output.rows(), batch, "Output rows mismatch");
        assert_eq!(output.cols(), seq_len * d_model, "Output cols mismatch");

        let (max_heads, dh) = parse_heads_from_params(params.len(), d_model);
        let head_param_count = 4 * dh * d_model + 3 * dh + d_model + 1;

        // Читаем h_raw и self_bias каждой головы из параметров.
        // params.parent_handle() — родительский буфер сегмента.
        let params_vec = self.download_gpu_handle_to_vec(params.parent_handle());
        let base = params.offset_elements();

        let h_raw_idx = base + max_heads * head_param_count;
        let h_raw = params_vec[h_raw_idx];
        let h_soft = compute_h_soft(h_raw, 1, max_heads);

        let total_in = batch * seq_len * d_model;
        let total_head = batch * seq_len * dh;
        let total_rt = batch * seq_len;

        // CPU-накопитель для взвешенного выхода.
        let mut y_accum = vec![0.0f32; total_in];

        let in_buf = self.get_gpu_subbuffer_from_handle(input);

        let mut heads_cache: Vec<GpuHeadCache> = Vec::new();

        for h in 0..max_heads {
            let w_h = head_weight(h_soft, h);
            if w_h <= HEAD_WEIGHT_EPS {
                continue;
            }

            // Смещения параметров головы.
            let hbase = base + h * head_param_count;
            let wq_off = hbase;
            let bq_off = wq_off + dh * d_model;
            let wk_off = bq_off + dh;
            let bk_off = wk_off + dh * d_model;
            let wv_off = bk_off + dh;
            let bv_off = wv_off + dh * d_model;
            let wo_off = bv_off + dh;
            let bo_off = wo_off + d_model * dh;
            let self_bias_off = bo_off + d_model;

            let self_bias = params_vec[self_bias_off];

            // Views на матрицы параметров головы.
            let parent = params.parent_handle().clone();
            let wq_view = MatrixBufferView::new(parent.clone(), wq_off, dh * d_model);
            let bq_view = MatrixBufferView::new(parent.clone(), bq_off, dh);
            let wk_view = MatrixBufferView::new(parent.clone(), wk_off, dh * d_model);
            let bk_view = MatrixBufferView::new(parent.clone(), bk_off, dh);
            let wv_view = MatrixBufferView::new(parent.clone(), wv_off, dh * d_model);
            let bv_view = MatrixBufferView::new(parent.clone(), bv_off, dh);
            let wo_view = MatrixBufferView::new(parent.clone(), wo_off, d_model * dh);
            let bo_view = MatrixBufferView::new(parent.clone(), bo_off, d_model);

            // Views внутрь caller-allocated буферов: per-head секции.
            let q_raw_view = MatrixBufferView::new(q_raw.clone(), h * total_head, total_head);
            let k_raw_view = MatrixBufferView::new(k_raw.clone(), h * total_head, total_head);
            let v_raw_view = MatrixBufferView::new(v_raw.clone(), h * total_head, total_head);
            let q_phi_view = MatrixBufferView::new(q_phi.clone(), h * total_head, total_head);
            let k_phi_view = MatrixBufferView::new(k_phi.clone(), h * total_head, total_head);
            let kv_view = MatrixBufferView::new(kv.clone(), h * dh * dh, dh * dh);
            let z_view = MatrixBufferView::new(z.clone(), h * dh, dh);

            // Subbuffers для шейдеров.
            let q_raw_sb = subbuffer_from_view(self, &q_raw_view);
            let k_raw_sb = subbuffer_from_view(self, &k_raw_view);
            let v_raw_sb = subbuffer_from_view(self, &v_raw_view);
            let q_phi_sb = subbuffer_from_view(self, &q_phi_view);
            let k_phi_sb = subbuffer_from_view(self, &k_phi_view);
            let kv_sb = subbuffer_from_view(self, &kv_view);
            let z_sb = subbuffer_from_view(self, &z_view);

            // ===== 1. QKV-проекции =====
            self.dispatch_linear_per_token(
                in_buf.clone(), &wq_view, &bq_view, q_raw_sb.clone(),
                batch, seq_len, d_model, dh,
            );
            self.dispatch_linear_per_token(
                in_buf.clone(), &wk_view, &bk_view, k_raw_sb.clone(),
                batch, seq_len, d_model, dh,
            );
            self.dispatch_linear_per_token(
                in_buf.clone(), &wv_view, &bv_view, v_raw_sb.clone(),
                batch, seq_len, d_model, dh,
            );

            // ===== 2. φ =====
            let phi_pipeline = &self.linear_attention_pipelines().phi;
            let push_phi = [total_head as u32];
            self.run_compute_shader(
                phi_pipeline,
                &[(0, q_raw_sb), (1, q_phi_sb.clone())],
                &push_phi,
                total_head,
            );
            self.run_compute_shader(
                phi_pipeline,
                &[(0, k_raw_sb), (1, k_phi_sb.clone())],
                &push_phi,
                total_head,
            );

            // ===== 3. compute_kvz =====
            let kvz_pipeline = &self.linear_attention_pipelines().compute_kvz;
            let push_kvz = [batch as u32, seq_len as u32, dh as u32];
            self.run_compute_shader(
                kvz_pipeline,
                &[
                    (0, k_phi_sb.clone()),
                    (1, v_raw_sb.clone()),
                    (2, kv_sb),
                    (3, z_sb),
                ],
                &push_kvz,
                dh * dh + dh,
            );

            // ===== 4. denom =====
            let denom_buf = self.allocate_gpu_matrix_handle(batch, seq_len);
            let denom_sb = self.get_gpu_subbuffer_from_handle(&denom_buf);
            let denom_pipeline = &self.linear_attention_pipelines().denom;
            let push_denom = [
                batch as u32,
                seq_len as u32,
                dh as u32,
                self_bias.to_bits(),
            ];
            self.run_compute_shader(
                denom_pipeline,
                &[
                    (0, q_phi_sb.clone()),
                    (1, subbuffer_from_view(self, &z_view)),
                    (2, denom_sb.clone()),
                ],
                &push_denom,
                total_rt,
            );

            // ===== 5. fwd: attn =====
            let attn_buf = self.allocate_gpu_matrix_handle(batch, seq_len * dh);
            let attn_sb = self.get_gpu_subbuffer_from_handle(&attn_buf);
            let fwd_pipeline = &self.linear_attention_pipelines().forward;
            let push_fwd = [
                batch as u32,
                seq_len as u32,
                dh as u32,
                self_bias.to_bits(),
            ];
            self.run_compute_shader(
                fwd_pipeline,
                &[
                    (0, q_phi_sb),
                    (1, subbuffer_from_view(self, &v_raw_view)),
                    (2, subbuffer_from_view(self, &kv_view)),
                    (3, denom_sb),
                    (4, attn_sb.clone()),
                ],
                &push_fwd,
                total_head,
            );

            // ===== 6. Выходная проекция: y_h = attn · Wo + bo =====
            let y_h_buf = self.allocate_gpu_matrix_handle(batch, seq_len * d_model);
            let y_h_sb = self.get_gpu_subbuffer_from_handle(&y_h_buf);
            self.dispatch_linear_per_token(
                attn_sb, &wo_view, &bo_view, y_h_sb,
                batch, seq_len, dh, d_model,
            );

            // ===== 7. Накопление y_accum += w_h · y_h =====
            let y_h_vec = self.download_gpu_handle_to_vec(&y_h_buf);
            for i in 0..total_in {
                y_accum[i] += w_h * y_h_vec[i];
            }

            heads_cache.push(GpuHeadCache {
                denom: denom_buf,
                attn: attn_buf,
                y_h: y_h_buf,
                weight: w_h,
                head_index: h,
            });
        }

        // Записываем итоговый взвешенный выход.
        self.copy_slice_to_gpu_handle(output, &y_accum);

        // Сохраняем кэш для backward.
        let key = cache_key(params, input);
        let cache = ForwardCache {
            heads: heads_cache,
            h_raw,
            h_soft,
            batch,
            seq_len,
            d_model,
        };
        FORWARD_CACHE.lock().unwrap().insert(key, cache);
    }

    // ------------------------------------------------------------------------
    // Multi-head backward.
    // ------------------------------------------------------------------------

    /// Обратный проход многоголового LinearAttention на GPU.
    ///
    /// Для каждой активной головы h:
    ///   1. dL/dw_h = dot(go, y_h)                            (CPU)
    ///   2. go_h = w_h · go                                   (CPU→GPU)
    ///   3. Запуск 12 backward-шейдеров, каждый пишет в свой
    ///      per-head участок grad_params
    ///   4. d_x_h скачивается и аккумулируется в gi_accum      (CPU)
    ///
    /// После цикла:
    ///   * gi_accum → grad_input
    ///   * dL/dh_raw = Σ_h dL/dw_h · dw_h/dh_soft · dh_soft/dh_raw → grad_params
    pub fn run_linear_attention_backward_buffered_handle_with_dims(
        &self,
        input: &MatrixBufferHandle,
        grad_out: &MatrixBufferHandle,
        params: &MatrixBufferView,
        grad_input: &MatrixBufferHandle,
        grad_params: &MatrixBufferView,
        seq_len: usize,
        d_model: usize,
        q_raw: &MatrixBufferHandle,
        k_raw: &MatrixBufferHandle,
        v_raw: &MatrixBufferHandle,
        q_phi: &MatrixBufferHandle,
        k_phi: &MatrixBufferHandle,
        kv: &MatrixBufferHandle,
        z: &MatrixBufferHandle,
    ) {
        assert!(input.is_gpu(), "Input handle must be GPU");
        assert!(grad_out.is_gpu(), "grad_out handle must be GPU");
        assert!(grad_input.is_gpu(), "grad_input handle must be GPU");
        assert!(grad_params.is_gpu(), "grad_params view must point to GPU buffer");
        assert!(params.is_gpu(), "Params view must point to GPU buffer");

        let batch = input.rows();
        assert_eq!(input.cols(), seq_len * d_model, "Input cols mismatch");
        assert_eq!(grad_out.rows(), batch);
        assert_eq!(grad_out.cols(), seq_len * d_model, "grad_out cols mismatch");
        assert_eq!(grad_input.rows(), batch);
        assert_eq!(grad_input.cols(), seq_len * d_model, "grad_input cols mismatch");

        let (max_heads, dh) = parse_heads_from_params(params.len(), d_model);
        let head_param_count = 4 * dh * d_model + 3 * dh + d_model + 1;

        assert_eq!(
            grad_params.len(),
            params.len(),
            "grad_params length must match params length"
        );

        // Извлекаем кэш, оставленный forward'ом.
        let key = cache_key(params, input);
        let cache = FORWARD_CACHE
            .lock()
            .unwrap()
            .remove(&key)
            .expect("LinearAttention GPU backward: forward cache not found");

        assert_eq!(cache.batch, batch, "Cache batch mismatch");
        assert_eq!(cache.seq_len, seq_len, "Cache seq_len mismatch");
        assert_eq!(cache.d_model, d_model, "Cache d_model mismatch");

        let h_raw = cache.h_raw;
        let h_soft = cache.h_soft;

        let base = params.offset_elements();
        let gbase = grad_params.offset_elements();

        // Скачиваем весь буфер параметров (нужны self_bias каждой головы).
        let params_vec = self.download_gpu_handle_to_vec(params.parent_handle());

        let total_in = batch * seq_len * d_model;
        let total_head = batch * seq_len * dh;
        let total_rt = batch * seq_len;

        // Скачиваем go один раз (переиспользуем для всех голов).
        let go_vec = self.download_gpu_handle_to_vec(grad_out);

        // CPU-накопитель для grad_input.
        let mut gi_accum = vec![0.0f32; total_in];

        // Производная h_soft по h_raw — одна на весь backward.
        let dh_soft_dh_raw = compute_h_soft_derivative(h_raw, 1, max_heads);

        // dL/dh_raw накапливается по головам.
        let mut dl_dh_raw_total = 0.0f32;

        // Переиспользуемые между головами промежуточные GPU-буферы.
        let go_h_buf = self.allocate_gpu_matrix_handle(batch, seq_len * d_model);
        let d_x_h_buf = self.allocate_gpu_matrix_handle(batch, seq_len * d_model);

        let d_attn_buf = self.allocate_gpu_matrix_handle(batch, seq_len * dh);
        let d_num_buf = self.allocate_gpu_matrix_handle(batch, seq_len * dh);
        let d_denom_buf = self.allocate_gpu_matrix_handle(batch, seq_len);
        let d_q_phi_buf = self.allocate_gpu_matrix_handle(batch, seq_len * dh);
        let d_z_buf = self.allocate_gpu_matrix_handle(dh, 1);
        let d_kv_buf = self.allocate_gpu_matrix_handle(dh, dh);
        let d_k_phi_buf = self.allocate_gpu_matrix_handle(batch, seq_len * dh);
        let d_v_buf = self.allocate_gpu_matrix_handle(batch, seq_len * dh);
        let d_q_raw_buf = self.allocate_gpu_matrix_handle(batch, seq_len * dh);
        let d_k_raw_buf = self.allocate_gpu_matrix_handle(batch, seq_len * dh);
        let d_self_bias_buf = self.allocate_gpu_matrix_handle(1, 1);

        let in_buf = self.get_gpu_subbuffer_from_handle(input);

        let pipelines = self.linear_attention_pipelines();

        for head_cache in &cache.heads {
            let h = head_cache.head_index;
            let w_h = head_cache.weight;
            let hbase = base + h * head_param_count;
            let ghbase = gbase + h * head_param_count;

            // Смещения в params.
            let wq_off = hbase;
            let wk_off = wq_off + dh * d_model + dh;
            let wv_off = wk_off + dh * d_model + dh;
            let wo_off = wv_off + dh * d_model + dh;
            let self_bias_off = wo_off + d_model * dh + d_model;

            // Смещения в grad_params.
            let gwq_off = ghbase;
            let gbq_off = gwq_off + dh * d_model;
            let gwk_off = gbq_off + dh;
            let gbk_off = gwk_off + dh * d_model;
            let gwv_off = gbk_off + dh;
            let gbv_off = gwv_off + dh * d_model;
            let gwo_off = gbv_off + dh;
            let gbo_off = gwo_off + d_model * dh;
            let gself_bias_off = gbo_off + d_model;

            let self_bias = params_vec[self_bias_off];

            // Views на матрицы параметров.
            let parent = params.parent_handle().clone();
            let wq_view = MatrixBufferView::new(parent.clone(), wq_off, dh * d_model);
            let wk_view = MatrixBufferView::new(parent.clone(), wk_off, dh * d_model);
            let wv_view = MatrixBufferView::new(parent.clone(), wv_off, dh * d_model);
            let wo_view = MatrixBufferView::new(parent.clone(), wo_off, d_model * dh);

            // Views на градиенты параметров.
            let gparent = grad_params.parent_handle().clone();
            let gwq_view = MatrixBufferView::new(gparent.clone(), gwq_off, dh * d_model);
            let gbq_view = MatrixBufferView::new(gparent.clone(), gbq_off, dh);
            let gwk_view = MatrixBufferView::new(gparent.clone(), gwk_off, dh * d_model);
            let gbk_view = MatrixBufferView::new(gparent.clone(), gbk_off, dh);
            let gwv_view = MatrixBufferView::new(gparent.clone(), gwv_off, dh * d_model);
            let gbv_view = MatrixBufferView::new(gparent.clone(), gbv_off, dh);
            let gwo_view = MatrixBufferView::new(gparent.clone(), gwo_off, d_model * dh);
            let gbo_view = MatrixBufferView::new(gparent.clone(), gbo_off, d_model);
            let gself_bias_view = MatrixBufferView::new(gparent.clone(), gself_bias_off, 1);

            // Per-head views внутрь caller-allocated q_raw, k_raw, ...
            let q_raw_view = MatrixBufferView::new(q_raw.clone(), h * total_head, total_head);
            let k_raw_view = MatrixBufferView::new(k_raw.clone(), h * total_head, total_head);
            let v_raw_view = MatrixBufferView::new(v_raw.clone(), h * total_head, total_head);
            let q_phi_view = MatrixBufferView::new(q_phi.clone(), h * total_head, total_head);
            let k_phi_view = MatrixBufferView::new(k_phi.clone(), h * total_head, total_head);
            let kv_view = MatrixBufferView::new(kv.clone(), h * dh * dh, dh * dh);
            let z_view = MatrixBufferView::new(z.clone(), h * dh, dh);

            let q_raw_sb = subbuffer_from_view(self, &q_raw_view);
            let k_raw_sb = subbuffer_from_view(self, &k_raw_view);
            let v_raw_sb = subbuffer_from_view(self, &v_raw_view);
            let q_phi_sb = subbuffer_from_view(self, &q_phi_view);
            let k_phi_sb = subbuffer_from_view(self, &k_phi_view);
            let kv_sb = subbuffer_from_view(self, &kv_view);
            let z_sb = subbuffer_from_view(self, &z_view);

            let wo_sb = subbuffer_from_view(self, &wo_view);
            let wq_sb = subbuffer_from_view(self, &wq_view);
            let wk_sb = subbuffer_from_view(self, &wk_view);
            let wv_sb = subbuffer_from_view(self, &wv_view);

            let d_attn_sb = self.get_gpu_subbuffer_from_handle(&d_attn_buf);
            let d_num_sb = self.get_gpu_subbuffer_from_handle(&d_num_buf);
            let d_denom_sb = self.get_gpu_subbuffer_from_handle(&d_denom_buf);
            let d_q_phi_sb = self.get_gpu_subbuffer_from_handle(&d_q_phi_buf);
            let d_z_sb = self.get_gpu_subbuffer_from_handle(&d_z_buf);
            let d_kv_sb = self.get_gpu_subbuffer_from_handle(&d_kv_buf);
            let d_k_phi_sb = self.get_gpu_subbuffer_from_handle(&d_k_phi_buf);
            let d_v_sb = self.get_gpu_subbuffer_from_handle(&d_v_buf);
            let d_q_raw_sb = self.get_gpu_subbuffer_from_handle(&d_q_raw_buf);
            let d_k_raw_sb = self.get_gpu_subbuffer_from_handle(&d_k_raw_buf);
            let d_self_bias_sb = self.get_gpu_subbuffer_from_handle(&d_self_bias_buf);

            // ===== 1. dL/dw_h = dot(go, y_h). =====
            let y_h_vec = self.download_gpu_handle_to_vec(&head_cache.y_h);
            let mut dl_dw_h = 0.0f32;
            for i in 0..total_in {
                dl_dw_h += go_vec[i] * y_h_vec[i];
            }

            // Вклад в dL/dh_raw.
            let dw_dhs = head_weight_derivative(h_soft, h);
            dl_dh_raw_total += dl_dw_h * dw_dhs;

            // ===== 2. go_h = w_h · go. =====
            let go_h_vec: Vec<f32> = go_vec.iter().map(|&g| w_h * g).collect();
            self.copy_slice_to_gpu_handle(&go_h_buf, &go_h_vec);
            let go_h_sb = self.get_gpu_subbuffer_from_handle(&go_h_buf);

            // ===== 3. bwd_dattn. =====
            let push_4dm = [batch as u32, seq_len as u32, dh as u32, d_model as u32];
            self.run_compute_shader(
                &pipelines.bwd_dattn,
                &[
                    (0, go_h_sb.clone()),
                    (1, wo_sb.clone()),
                    (2, d_attn_sb.clone()),
                ],
                &push_4dm,
                total_head,
            );

            // ===== 4. bwd_grad_wo. =====
            self.run_compute_shader(
                &pipelines.bwd_grad_wo,
                &[
                    (0, go_h_sb.clone()),
                    (1, self.get_gpu_subbuffer_from_handle(&head_cache.attn)),
                    (2, subbuffer_from_view(self, &gwo_view)),
                ],
                &push_4dm,
                d_model * dh,
            );

            // ===== 5. bwd_grad_bo. =====
            let push_3d = [batch as u32, seq_len as u32, d_model as u32];
            self.run_compute_shader(
                &pipelines.bwd_grad_bo,
                &[
                    (0, go_h_sb.clone()),
                    (1, subbuffer_from_view(self, &gbo_view)),
                ],
                &push_3d,
                d_model,
            );

            // ===== 6. bwd_dnum_denom. =====
            let push_3dh = [batch as u32, seq_len as u32, dh as u32];
            self.run_compute_shader(
                &pipelines.bwd_dnum_denom,
                &[
                    (0, d_attn_sb.clone()),
                    (1, self.get_gpu_subbuffer_from_handle(&head_cache.denom)),
                    (2, self.get_gpu_subbuffer_from_handle(&head_cache.attn)),
                    (3, d_num_sb.clone()),
                    (4, d_denom_sb.clone()),
                ],
                &push_3dh,
                total_rt,
            );

            // ===== 7. bwd_dq_phi. =====
            self.run_compute_shader(
                &pipelines.bwd_dq_phi,
                &[
                    (0, d_denom_sb.clone()),
                    (1, z_sb.clone()),
                    (2, d_num_sb.clone()),
                    (3, kv_sb.clone()),
                    (4, d_q_phi_sb.clone()),
                ],
                &push_3dh,
                total_head,
            );

            // ===== 8. bwd_dz. =====
            self.run_compute_shader(
                &pipelines.bwd_dz,
                &[
                    (0, d_denom_sb.clone()),
                    (1, q_phi_sb.clone()),
                    (2, d_z_sb.clone()),
                ],
                &push_3dh,
                dh,
            );

            // ===== 9. bwd_dkv. =====
            self.run_compute_shader(
                &pipelines.bwd_dkv,
                &[
                    (0, d_num_sb.clone()),
                    (1, q_phi_sb.clone()),
                    (2, d_kv_sb.clone()),
                ],
                &push_3dh,
                dh * dh,
            );

            // ===== 10. bwd_dk_phi. =====
            self.run_compute_shader(
                &pipelines.bwd_dk_phi,
                &[
                    (0, d_kv_sb.clone()),
                    (1, v_raw_sb.clone()),
                    (2, d_z_sb.clone()),
                    (3, d_k_phi_sb.clone()),
                ],
                &push_3dh,
                total_head,
            );

            // ===== 11. bwd_dv. =====
            let push_4dh = [
                batch as u32,
                seq_len as u32,
                dh as u32,
                self_bias.to_bits(),
            ];
            self.run_compute_shader(
                &pipelines.bwd_dv,
                &[
                    (0, d_num_sb.clone()),
                    (1, d_kv_sb.clone()),
                    (2, k_phi_sb.clone()),
                    (3, d_v_sb.clone()),
                ],
                &push_4dh,
                total_head,
            );

            // ===== 12. bwd_dqk_raw. =====
            self.run_compute_shader(
                &pipelines.bwd_dqk_raw,
                &[
                    (0, d_q_phi_sb.clone()),
                    (1, q_raw_sb.clone()),
                    (2, d_q_raw_sb.clone()),
                    (3, d_k_phi_sb.clone()),
                    (4, k_raw_sb.clone()),
                    (5, d_k_raw_sb.clone()),
                ],
                &push_3dh,
                total_head,
            );

            // ===== 13. bwd_grad_qkv_weights. =====
            self.run_compute_shader(
                &pipelines.bwd_grad_qkv_weights,
                &[
                    (0, d_q_raw_sb.clone()),
                    (1, d_k_raw_sb.clone()),
                    (2, d_v_sb.clone()),
                    (3, in_buf.clone()),
                    (4, subbuffer_from_view(self, &gwq_view)),
                    (5, subbuffer_from_view(self, &gwk_view)),
                    (6, subbuffer_from_view(self, &gwv_view)),
                ],
                &push_4dm,
                dh * d_model,
            );

            // ===== 14. bwd_grad_qkv_bias. =====
            self.run_compute_shader(
                &pipelines.bwd_grad_qkv_bias,
                &[
                    (0, d_q_raw_sb.clone()),
                    (1, d_k_raw_sb.clone()),
                    (2, d_v_sb.clone()),
                    (3, subbuffer_from_view(self, &gbq_view)),
                    (4, subbuffer_from_view(self, &gbk_view)),
                    (5, subbuffer_from_view(self, &gbv_view)),
                ],
                &push_3dh,
                dh,
            );

            // ===== 15. bwd_self_bias (атомарное накопление). =====
            // Обнуляем элемент grad_params[gself_bias_off] перед запуском.
            let zero_one = self.upload_vec_to_gpu_handle(&[0.0f32], 1, 1);
            self.copy_gpu_handle_region(
                &zero_one,
                grad_params.parent_handle(),
                0,
                gself_bias_off,
                1,
            );

            self.run_compute_shader(
                &pipelines.bwd_self_bias,
                &[
                    (0, d_denom_sb.clone()),
                    (1, d_num_sb.clone()),
                    (2, v_raw_sb.clone()),
                    (3, subbuffer_from_view(self, &gself_bias_view)),
                ],
                &push_3dh,
                total_rt,
            );

            // ===== 16. bwd_dx. =====
            let d_x_h_sb = self.get_gpu_subbuffer_from_handle(&d_x_h_buf);
            self.run_compute_shader(
                &pipelines.bwd_dx,
                &[
                    (0, d_q_raw_sb),
                    (1, d_k_raw_sb),
                    (2, d_v_sb),
                    (3, wq_sb),
                    (4, wk_sb),
                    (5, wv_sb),
                    (6, d_x_h_sb.clone()),
                ],
                &push_4dm,
                total_in,
            );

            // ===== 17. gi_accum += d_x_h =====
            let d_x_h_vec = self.download_gpu_handle_to_vec(&d_x_h_buf);
            for i in 0..total_in {
                gi_accum[i] += d_x_h_vec[i];
            }
        }

        // Записываем итоговый градиент по входу.
        self.copy_slice_to_gpu_handle(grad_input, &gi_accum);

        // Вычисляем dL/dh_raw и записываем в grad_params[h_raw_idx].
        let dl_dh_raw = dl_dh_raw_total * dh_soft_dh_raw;
        let h_raw_grad_idx = gbase + max_heads * head_param_count;

        let h_raw_grad_handle = self.upload_vec_to_gpu_handle(&[dl_dh_raw], 1, 1);
        self.copy_gpu_handle_region(
            &h_raw_grad_handle,
            grad_params.parent_handle(),
            0,
            h_raw_grad_idx,
            1,
        );

        // Temp-буферы освобождаются автоматически при выходе из функции.
        // `cache` уже удалён из FORWARD_CACHE, его handles освободятся здесь.
        drop(cache);
    }
}