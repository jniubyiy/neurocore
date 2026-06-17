// ============================================================
// Файл: src/model_plan/plan.rs
// ============================================================

use super::blueprint::LayerBlueprint;
use crate::model_plan::dim::Dim;
use crate::model_plan::param_store::{ParamSlice, ParamStore};

pub struct BuiltModel1D {
    pub layers: Vec<Box<dyn crate::layers::Layer + Send + Sync>>,
    pub slices: Vec<ParamSlice>,
    pub store: ParamStore,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl BuiltModel1D {
    pub fn into_single_model(self) -> crate::dispatchers::single::SingleModel1D {
        let layers: Vec<Box<dyn crate::layers::Layer>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer>)
            .collect();
        crate::dispatchers::single::SingleModel1D::new(layers, self.slices, self.store)
    }

    pub fn into_auto_model(self, num_threads: usize) -> crate::dispatchers::auto::AutoModel1D {
        let layers: Vec<Box<dyn crate::layers::Layer>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer>)
            .collect();
        crate::dispatchers::auto::AutoModel1D::new(layers, self.slices, self.store, num_threads)
    }

    pub fn into_trained_model(self, num_threads: usize) -> crate::dispatchers::trained::TrainedModel1D {
        let layers: Vec<Box<dyn crate::layers::Layer>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer>)
            .collect();
        crate::dispatchers::trained::TrainedModel1D::new(layers, self.slices, self.store, num_threads)
    }
}

pub struct BuiltModel2D {
    pub layers: Vec<Box<dyn crate::layers::Layer2D + Send + Sync>>,
    pub slices: Vec<ParamSlice>,
    pub store: ParamStore,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl BuiltModel2D {
    pub fn into_single_model(self) -> crate::dispatchers::single::SingleModel2D {
        let layers: Vec<Box<dyn crate::layers::Layer2D>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer2D>)
            .collect();
        crate::dispatchers::single::SingleModel2D::new(layers, self.slices, self.store, 1)
    }

    pub fn into_auto_model(self, num_threads: usize) -> crate::dispatchers::auto::AutoModel2D {
        let layers: Vec<Box<dyn crate::layers::Layer2D>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer2D>)
            .collect();
        crate::dispatchers::auto::AutoModel2D::new(layers, self.slices, self.store, num_threads)
    }

    pub fn into_trained_model(self, num_threads: usize) -> crate::dispatchers::trained::TrainedModel2D {
        let layers: Vec<Box<dyn crate::layers::Layer2D>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer2D>)
            .collect();
        crate::dispatchers::trained::TrainedModel2D::new(layers, self.slices, self.store, num_threads)
    }
}

pub struct BuiltModel3D {
    pub layers: Vec<Box<dyn crate::layers::Layer3D + Send + Sync>>,
    pub slices: Vec<ParamSlice>,
    pub store: ParamStore,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl BuiltModel3D {
    pub fn into_single_model(self) -> crate::dispatchers::single::SingleModel3D {
        let layers: Vec<Box<dyn crate::layers::Layer3D>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer3D>)
            .collect();
        crate::dispatchers::single::SingleModel3D::new(layers, self.slices, self.store, 1, 1)
    }

    pub fn into_auto_model(self, num_threads: usize) -> crate::dispatchers::auto::AutoModel3D {
        let layers: Vec<Box<dyn crate::layers::Layer3D>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer3D>)
            .collect();
        crate::dispatchers::auto::AutoModel3D::new(layers, self.slices, self.store, num_threads)
    }

    pub fn into_trained_model(self, num_threads: usize) -> crate::dispatchers::trained::TrainedModel3D {
        let layers: Vec<Box<dyn crate::layers::Layer3D>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer3D>)
            .collect();
        crate::dispatchers::trained::TrainedModel3D::new(layers, self.slices, self.store, num_threads)
    }
}

pub struct BuiltModel4D {
    pub layers: Vec<Box<dyn crate::layers::Layer4D + Send + Sync>>,
    pub slices: Vec<ParamSlice>,
    pub store: ParamStore,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl BuiltModel4D {
    pub fn into_single_model(self) -> crate::dispatchers::single::SingleModel4D {
        let layers: Vec<Box<dyn crate::layers::Layer4D>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer4D>)
            .collect();
        crate::dispatchers::single::SingleModel4D::new(layers, self.slices, self.store, 1, 1, 1)
    }

    pub fn into_auto_model(self, num_threads: usize) -> crate::dispatchers::auto::AutoModel4D {
        let layers: Vec<Box<dyn crate::layers::Layer4D>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer4D>)
            .collect();
        crate::dispatchers::auto::AutoModel4D::new(layers, self.slices, self.store, num_threads)
    }

    pub fn into_trained_model(self, num_threads: usize) -> crate::dispatchers::trained::TrainedModel4D {
        let layers: Vec<Box<dyn crate::layers::Layer4D>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer4D>)
            .collect();
        crate::dispatchers::trained::TrainedModel4D::new(layers, self.slices, self.store, num_threads)
    }
}

pub struct BuiltModel5D {
    pub layers: Vec<Box<dyn crate::layers::Layer5D + Send + Sync>>,
    pub slices: Vec<ParamSlice>,
    pub store: ParamStore,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl BuiltModel5D {
    pub fn into_single_model(self) -> crate::dispatchers::single::SingleModel5D {
        let layers: Vec<Box<dyn crate::layers::Layer5D>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer5D>)
            .collect();
        crate::dispatchers::single::SingleModel5D::new(layers, self.slices, self.store, 1, 1, 1, 1)
    }

    pub fn into_auto_model(self, num_threads: usize) -> crate::dispatchers::auto::AutoModel5D {
        let layers: Vec<Box<dyn crate::layers::Layer5D>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer5D>)
            .collect();
        crate::dispatchers::auto::AutoModel5D::new(layers, self.slices, self.store, num_threads)
    }

    pub fn into_trained_model(self, num_threads: usize) -> crate::dispatchers::trained::TrainedModel5D {
        let layers: Vec<Box<dyn crate::layers::Layer5D>> = self.layers
            .into_iter()
            .map(|b| b as Box<dyn crate::layers::Layer5D>)
            .collect();
        crate::dispatchers::trained::TrainedModel5D::new(layers, self.slices, self.store, num_threads)
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
            if !super::blueprint::is_power_of_two(bp.in_features) {
                return Err(format!(
                    "Слой {}: in_features ({}) должен быть степенью двойки (1,2,4,8,…)",
                    i, bp.in_features
                ));
            }
            if !super::blueprint::is_power_of_two(bp.out_features) {
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
            if bp.kind == super::blueprint::LayerKind::Softmax && i != blueprints.len() - 1 {
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

    pub fn build_3d(&self) -> BuiltModel3D {
        for bp in &self.blueprints {
            assert_eq!(bp.dim, Dim::Dim3, "Plan::build_3d: все слои должны быть Dim3");
        }
        let mut store = ParamStore::new();
        let mut layers = Vec::new();
        let mut slices = Vec::new();
        for bp in &self.blueprints {
            let (layer, slice) = bp.build_layer_3d(&mut store);
            layers.push(layer);
            slices.push(slice);
        }
        let input_dim = self.blueprints.first().unwrap().in_features;
        let output_dim = self.blueprints.last().unwrap().out_features;
        BuiltModel3D { layers, slices, store, input_dim, output_dim }
    }

    pub fn build_4d(&self) -> BuiltModel4D {
        for bp in &self.blueprints {
            assert_eq!(bp.dim, Dim::Dim4, "Plan::build_4d: все слои должны быть Dim4");
        }
        let mut store = ParamStore::new();
        let mut layers = Vec::new();
        let mut slices = Vec::new();
        for bp in &self.blueprints {
            let (layer, slice) = bp.build_layer_4d(&mut store);
            layers.push(layer);
            slices.push(slice);
        }
        let input_dim = self.blueprints.first().unwrap().in_features;
        let output_dim = self.blueprints.last().unwrap().out_features;
        BuiltModel4D { layers, slices, store, input_dim, output_dim }
    }

    pub fn build_5d(&self) -> BuiltModel5D {
        for bp in &self.blueprints {
            assert_eq!(bp.dim, Dim::Dim5, "Plan::build_5d: все слои должны быть Dim5");
        }
        let mut store = ParamStore::new();
        let mut layers = Vec::new();
        let mut slices = Vec::new();
        for bp in &self.blueprints {
            let (layer, slice) = bp.build_layer_5d(&mut store);
            layers.push(layer);
            slices.push(slice);
        }
        let input_dim = self.blueprints.first().unwrap().in_features;
        let output_dim = self.blueprints.last().unwrap().out_features;
        BuiltModel5D { layers, slices, store, input_dim, output_dim }
    }

    pub fn input_dim(&self) -> usize { self.blueprints.first().unwrap().in_features }
    pub fn output_dim(&self) -> usize { self.blueprints.last().unwrap().out_features }
}