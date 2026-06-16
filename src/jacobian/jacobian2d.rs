use crate::jacobian::Jacobian;

#[derive(Debug, Clone)]
pub struct Jacobian2D {
    pub rows: usize,
    pub out_features: usize,
    pub num_params: usize,
    pub data: Vec<Vec<Vec<f32>>>,
}

impl Jacobian2D {
    /// `rows` – число примеров в батче,
    /// `cols` – число выходных признаков (out_features),
    /// `params` – общее число параметров.
    pub fn new(rows: usize, cols: usize, params: usize) -> Self {
        Jacobian2D {
            rows,
            out_features: cols,
            num_params: params,
            data: vec![vec![vec![0.0; params]; cols]; rows],
        }
    }

    pub fn row_jacobian(&self, r: usize) -> Jacobian {
        let mut j = Jacobian::new(self.out_features, self.num_params);
        for c in 0..self.out_features {
            for p in 0..self.num_params {
                j.data[c][p] = self.data[r][c][p];
            }
        }
        j
    }

    pub fn set_row_jacobian(&mut self, r: usize, j: &Jacobian) {
        assert_eq!(j.out_features, self.out_features);
        assert_eq!(j.num_params, self.num_params);
        for c in 0..self.out_features {
            for p in 0..self.num_params {
                self.data[r][c][p] = j.data[c][p];
            }
        }
    }
}





