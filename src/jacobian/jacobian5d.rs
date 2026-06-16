use crate::jacobian::Jacobian4D;

#[derive(Debug, Clone)]
pub struct Jacobian5D {
    pub outer: usize,
    pub dim1: usize,
    pub depth: usize,
    pub rows: usize,
    pub out_features: usize,
    pub num_params: usize,
    pub data: Vec<Vec<Vec<Vec<Vec<Vec<f32>>>>>>,
}

impl Jacobian5D {
    pub fn new(outer: usize, dim1: usize, depth: usize, rows: usize, cols: usize, params: usize) -> Self {
        Jacobian5D {
            outer,
            dim1,
            depth,
            rows,
            out_features: cols,
            num_params: params,
            data: vec![vec![vec![vec![vec![vec![0.0; params]; cols]; rows]; depth]; dim1]; outer],
        }
    }

    pub fn slice_jacobian(&self, o: usize) -> Jacobian4D {
        let mut j4d = Jacobian4D::new(self.dim1, self.depth, self.rows, self.out_features, self.num_params);
        for d1 in 0..self.dim1 {
            for d in 0..self.depth {
                for r in 0..self.rows {
                    for c in 0..self.out_features {
                        for p in 0..self.num_params {
                            j4d.data[d1][d][r][c][p] = self.data[o][d1][d][r][c][p];
                        }
                    }
                }
            }
        }
        j4d
    }

    pub fn set_slice_jacobian(&mut self, o: usize, j4d: &Jacobian4D) {
        assert_eq!(j4d.dim1, self.dim1);
        assert_eq!(j4d.depth, self.depth);
        assert_eq!(j4d.rows, self.rows);
        assert_eq!(j4d.out_features, self.out_features);
        assert_eq!(j4d.num_params, self.num_params);
        for d1 in 0..self.dim1 {
            for d in 0..self.depth {
                for r in 0..self.rows {
                    for c in 0..self.out_features {
                        for p in 0..self.num_params {
                            self.data[o][d1][d][r][c][p] = j4d.data[d1][d][r][c][p];
                        }
                    }
                }
            }
        }
    }
}





