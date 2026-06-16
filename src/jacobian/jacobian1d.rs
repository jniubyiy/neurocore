#[derive(Debug, Clone)]
pub struct Jacobian {
    pub out_features: usize,
    pub num_params: usize,
    pub data: Vec<Vec<f32>>,
}

impl Jacobian {
    /// `cols` – число выходных нейронов (out_features),
    /// `params` – общее количество параметров модели.
    pub fn new(cols: usize, params: usize) -> Self {
        Jacobian {
            out_features: cols,
            num_params: params,
            data: vec![vec![0.0; params]; cols],
        }
    }

    pub fn rows(&self) -> usize {
        self.out_features
    }

    pub fn cols(&self) -> usize {
        self.num_params
    }
}





