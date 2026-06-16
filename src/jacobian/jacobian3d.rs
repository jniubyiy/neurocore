use crate::jacobian::Jacobian2D;

#[derive(Debug, Clone)]
pub struct Jacobian3D {
    pub depth: usize,
    pub rows: usize,
    pub out_features: usize,
    pub num_params: usize,
    pub data: Vec<Vec<Vec<Vec<f32>>>>,
}

impl Jacobian3D {
    pub fn new(depth: usize, rows: usize, cols: usize, params: usize) -> Self {
        Jacobian3D {
            depth,
            rows,
            out_features: cols,
            num_params: params,
            data: vec![vec![vec![vec![0.0; params]; cols]; rows]; depth],
        }
    }

    pub fn slice_jacobian(&self, d: usize) -> Jacobian2D {
        let mut j2d = Jacobian2D::new(self.rows, self.out_features, self.num_params);
        for r in 0..self.rows {
            for c in 0..self.out_features {
                for p in 0..self.num_params {
                    j2d.data[r][c][p] = self.data[d][r][c][p];
                }
            }
        }
        j2d
    }

    pub fn set_slice_jacobian(&mut self, d: usize, j2d: &Jacobian2D) {
        assert_eq!(j2d.rows, self.rows);
        assert_eq!(j2d.out_features, self.out_features);
        assert_eq!(j2d.num_params, self.num_params);
        for r in 0..self.rows {
            for c in 0..self.out_features {
                for p in 0..self.num_params {
                    self.data[d][r][c][p] = j2d.data[r][c][p];
                }
            }
        }
    }
}





