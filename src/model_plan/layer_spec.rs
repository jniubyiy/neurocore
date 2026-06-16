use crate::dispatchers::common::{LayerType, LayerInfo};

/// Универсальное описание одного слоя.
pub struct LayerSpec {
    pub layer_type: LayerType,
    pub in_features: usize,
    pub out_features: usize,
}

impl LayerSpec {
    pub fn linear(in_features: usize, out_features: usize) -> Self {
        Self { layer_type: LayerType::Linear, in_features, out_features }
    }
    pub fn activation(act: LayerType, size: usize) -> Self {
        Self { layer_type: act, in_features: size, out_features: size }
    }

    /// Превратить в `LayerInfo` для заданного размера батча.
    pub fn to_info(&self, total_rows: usize) -> LayerInfo {
        LayerInfo {
            id: 0, // будет переопределено в плане
            layer_type: self.layer_type.clone(),
            in_features: self.in_features,
            out_features: self.out_features,
            total_rows,
        }
    }
}