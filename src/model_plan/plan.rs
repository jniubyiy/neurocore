// src/model_plan/plan.rs

use super::blueprint::LayerBlueprint;
use crate::dispatchers::auto_model::MixedModel;

pub struct Plan {
    blueprints: Vec<LayerBlueprint>,
}

impl Plan {
    /// Создаёт план из готовых `LayerBlueprint`.
    pub fn new(blueprints: Vec<LayerBlueprint>) -> Result<Self, String> {
        if blueprints.is_empty() {
            return Err("План не может быть пустым".into());
        }

        for (i, bp) in blueprints.iter().enumerate() {
            if bp.in_features.is_empty() || bp.out_features.is_empty() {
                return Err(format!("Слой {}: размерности не могут быть пустыми", i));
            }
            if bp.in_features.len() != 1 || bp.out_features.len() != 1 {
                return Err(format!(
                    "Слой {} ({} → {}) имеет нестандартное число входов/выходов. \
                     Мультивариабельные слои пока не поддерживаются.",
                    i, bp.in_features.len(), bp.out_features.len()
                ));
            }
            if bp.in_features[0] == 0 || bp.out_features[0] == 0 {
                return Err(format!("Слой {}: размерности не могут быть нулевыми", i));
            }
        }

        // Проверка совместимости размерностей и размеров последней оси
        for i in 1..blueprints.len() {
            let prev = &blueprints[i - 1];
            let curr = &blueprints[i];
            if prev.output_dim != curr.input_dim {
                return Err(format!(
                    "Несовместимость размерностей между слоем {} (выходная {:?}) и слоем {} (входная {:?})",
                    i, prev.output_dim, i + 1, curr.input_dim
                ));
            }
            if prev.out_features.len() == 1 && curr.in_features.len() == 1 {
                let prev_out = prev.out_features[0];
                let curr_in = curr.in_features[0];
                if prev_out != curr_in {
                    return Err(format!(
                        "Несовместимость размеров последней оси между слоем {} (выход {}) и слоем {} (вход {})",
                        i, prev_out, i + 1, curr_in
                    ));
                }
            }
        }

        // Softmax допускается только на последнем слое
        for (i, bp) in blueprints.iter().enumerate() {
            if bp.kind == super::blueprint::LayerKind::Softmax && i != blueprints.len() - 1 {
                return Err("Softmax допускается только на последнем слое".into());
            }
        }

        Ok(Plan { blueprints })
    }

    /// Создаёт план из списка описаний слоёв `LayerDesc`.
    pub fn from_descs(descs: Vec<super::layer_desc::LayerDesc>) -> Result<Self, String> {
        let blueprints: Vec<LayerBlueprint> = descs.into_iter().map(|d| d.into_blueprint()).collect();
        Plan::new(blueprints)
    }

    /// Собирает модель для смешанных размерностей (всегда успешно, т.к. план проверен).
    pub fn build(&self) -> MixedModel {
        MixedModel::from_plan(self.blueprints.clone(), 1).expect("Plan уже проверен")
    }

    pub fn input_dim1(&self) -> usize {
        self.blueprints.first().unwrap().in_features[0]
    }

    pub fn output_dim1(&self) -> usize {
        self.blueprints.last().unwrap().out_features[0]
    }
}