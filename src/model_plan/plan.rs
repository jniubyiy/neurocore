use super::blueprint::{LayerBlueprint, LayerKind, is_power_of_two};
use crate::model_plan::dim::Dim;
use crate::model_plan::param_store::ParamStore;
use std::sync::Arc;

pub struct BuiltModel1D {
    pub layers: Vec<Arc<dyn crate::layers::Layer + Send + Sync>>,
    pub slices: Vec<crate::model_plan::param_store::ParamSlice>,
    pub store: ParamStore,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl BuiltModel1D {
    pub fn into_single_model(self) -> crate::dispatchers::single::SingleModel1D {
        crate::dispatchers::single::SingleModel1D::new(self.layers, self.slices, self.store)
    }
    pub fn into_auto_model(self, num_threads: usize) -> crate::dispatchers::auto::AutoModel1D {
        crate::dispatchers::auto::AutoModel1D::new(self.layers, self.slices, self.store, num_threads)
    }
    pub fn into_trained_model(self, num_threads: usize) -> crate::dispatchers::trained::TrainedModel1D {
        crate::dispatchers::trained::TrainedModel1D::new(self.layers, self.slices, self.store, num_threads)
    }
}

pub struct BuiltModel2D {
    pub layers: Vec<Arc<dyn crate::layers::Layer2D + Send + Sync>>,
    pub slices: Vec<crate::model_plan::param_store::ParamSlice>,
    pub store: ParamStore,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl BuiltModel2D {
    pub fn into_single_model(self) -> crate::dispatchers::single::SingleModel2D {
        crate::dispatchers::single::SingleModel2D::new(self.layers, self.slices, self.store)
    }
    pub fn into_auto_model(self, num_threads: usize) -> crate::dispatchers::auto::AutoModel2D {
        crate::dispatchers::auto::AutoModel2D::new(self.layers, self.slices, self.store, num_threads)
    }
    pub fn into_trained_model(self, num_threads: usize) -> crate::dispatchers::trained::TrainedModel2D {
        crate::dispatchers::trained::TrainedModel2D::new(self.layers, self.slices, self.store, num_threads)
    }
}

pub struct Plan {
    blueprints: Vec<LayerBlueprint>,
}

impl Plan {
    pub fn new(blueprints: Vec<LayerBlueprint>) -> Result<Self, String> {
        if blueprints.is_empty() {
            return Err("План не может быть пустым".into());
        }

        for (i, bp) in blueprints.iter().enumerate() {
            if bp.in_features == 0 || bp.out_features == 0 {
                return Err(format!("Слой {}: размерности не могут быть нулевыми", i));
            }
            if !is_power_of_two(bp.in_features) {
                return Err(format!(
                    "Слой {}: in_features ({}) должен быть степенью двойки (1,2,4,8,…)",
                    i, bp.in_features
                ));
            }
            if !is_power_of_two(bp.out_features) {
                return Err(format!(
                    "Слой {}: out_features ({}) должен быть степенью двойки",
                    i, bp.out_features
                ));
            }
        }

        for i in 1..blueprints.len() {
            let prev_out = blueprints[i - 1].out_features;
            let curr_in = blueprints[i].in_features;
            if prev_out != curr_in {
                return Err(format!(
                    "Несовместимость размеров между слоем {} (выход {}) и слоем {} (вход {})",
                    i, prev_out, i + 1, curr_in
                ));
            }
        }

        for (i, bp) in blueprints.iter().enumerate() {
            if bp.kind == LayerKind::Softmax && i != blueprints.len() - 1 {
                return Err("Softmax допускается только на последнем слое".into());
            }
        }

        Ok(Plan { blueprints })
    }

    pub fn build_1d(&self) -> BuiltModel1D {
        for bp in &self.blueprints {
            assert_eq!(bp.dim, Dim::Dim1, "Plan::build_1d: все слои должны быть Dim1");
        }
        let mut store = ParamStore::new();
        let mut layers = Vec::new();
        let mut slices = Vec::new();
        for bp in &self.blueprints {
            let (layer, slice) = bp.build_layer_1d(&mut store);
            layers.push(layer);
            slices.push(slice);
        }
        let input_dim = self.blueprints.first().unwrap().in_features;
        let output_dim = self.blueprints.last().unwrap().out_features;
        BuiltModel1D { layers, slices, store, input_dim, output_dim }
    }

    pub fn build_2d(&self) -> BuiltModel2D {
        for bp in &self.blueprints {
            assert_eq!(bp.dim, Dim::Dim2, "Plan::build_2d: все слои должны быть Dim2");
        }
        let mut store = ParamStore::new();
        let mut layers = Vec::new();
        let mut slices = Vec::new();
        for bp in &self.blueprints {
            let (layer, slice) = bp.build_layer_2d(&mut store);
            layers.push(layer);
            slices.push(slice);
        }
        let input_dim = self.blueprints.first().unwrap().in_features;
        let output_dim = self.blueprints.last().unwrap().out_features;
        BuiltModel2D { layers, slices, store, input_dim, output_dim }
    }

    pub fn input_dim(&self) -> usize { self.blueprints.first().unwrap().in_features }
    pub fn output_dim(&self) -> usize { self.blueprints.last().unwrap().out_features }
}