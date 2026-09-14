// src/layers/feature_fusion/feature_fusion.rs

use crate::layers::UniversalLayer;

/// Слой FeatureFusion — обучаемое глобальное агрегирование признаков
/// через softmax-attention с обучаемой температурой.
///
/// # Формула
/// Для каждого выхода `j ∈ [0, fout)`:
///
///   w_{j,i}   = softmax_i( L_{j,:} / T_j )
///   u_{j,r}   = Σ_i w_{j,i} · x_{i,r}
///   y_{j,r}   = u_{j,r}                                (bias отсутствует)
///
/// где:
/// * `L`     — обучаемые логиты размера `fout × fin`;
/// * `T_raw` — обучаемая температура (по одной на выход j);
/// * `T_eff` = `|T_raw| + 1e-6` — гарантирует положительность и конечность
///   `1/T`;
/// * `w_{j,:}` — нормированные веса внимания выхода j.
///
/// # Отличие от torch-версии (concat + Linear)
///
/// В torch слияние признаков делается через `Linear(concat)`: веса свободные,
/// любого знака, любой величины, без нормировки. Это даёт максимальную
/// выразительность, но теряет интерпретируемость и допускает «схлопывание»
/// на один доминирующий признак с гигантским весом.
///
/// Здесь веса проходят через softmax: `w_{j,i} ≥ 0`, `Σ_i w_{j,i} = 1`.
/// Это:
/// * даёт прямую интерпретацию «насколько выход j доверяет признаку i»;
/// * регуляризует вклад признаков естественным образом;
/// * исключает доминирование одного признака.
///
/// # Обучаемая температура
///
/// `T_raw[j]` позволяет слою самому подбирать резкость распределения весов:
/// * при малой `T_eff[j]` softmax «заостряется» → выход j почти копирует
///   один входной признак (выбор one-hot);
/// * при большой `T_eff[j]` softmax «размывается» → выход j становится
///   усреднением по всем признакам.
///
/// Инициализация `T_raw ≈ 1` даёт поведение, эквивалентное обычному
/// softmax без температуры. Всё остальное — на усмотрение обучения.
///
/// # Отличие от предыдущей версии (с bias)
///
/// Bias убран. Причина: при инициализации `L ≈ 1` softmax даёт веса ≈ 1/fin,
/// вклад признаков в выход мал (≈ `mean(x) / fin`), а bias с init ≈ 1.0
/// начинал доминировать. Слой превращался в «константу + слабый сигнал» и
/// служил одной из причин плато при обучении. Без bias слой честно
/// агрегирует входы и передаёт их дальше.
///
/// # Параметры
///
/// Порядок в общем буфере параметров (смещение `slice.start`):
///
/// | Смещение          | Размер       | Что                     |
/// |-------------------|--------------|-------------------------|
/// | `0`               | `fout · fin` | логиты `L[j·fin + i]`   |
/// | `fout · fin`      | `fout`       | температуры `T_raw[j]`  |
///
/// Общее число параметров: `fout·fin + fout = fout·(fin + 1)` — то же, что
/// и в версии с bias.
pub struct FeatureFusion {
    pub in_features: usize,
    pub out_features: usize,
}

impl FeatureFusion {
    /// Создаёт слой.
    ///
    /// # Паника
    /// Паникует, если `in_features == 0` или `out_features == 0`.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        assert!(in_features > 0, "FeatureFusion: in_features must be positive");
        assert!(out_features > 0, "FeatureFusion: out_features must be positive");
        Self { in_features, out_features }
    }
}

impl UniversalLayer for FeatureFusion {
    fn as_feature_fusion(&self) -> Option<&FeatureFusion> {
        Some(self)
    }

    fn param_len(&self) -> usize {
        self.out_features * (self.in_features + 1)
    }

    fn input_features(&self) -> usize {
        self.in_features
    }

    fn output_features(&self) -> usize {
        self.out_features
    }
}