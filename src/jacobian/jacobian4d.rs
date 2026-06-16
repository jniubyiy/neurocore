use crate::jacobian::Jacobian3D;

#[derive(Debug, Clone)]
pub struct Jacobian4D {
    pub dim1: usize,
    pub depth: usize,
    pub rows: usize,
    pub out_features: usize,
    pub num_params: usize,
    pub data: Vec<Vec<Vec<Vec<Vec<f32>>>>>,
}

impl Jacobian4D {
    pub fn new(dim1: usize, depth: usize, rows: usize, cols: usize, params: usize) -> Self {
        Jacobian4D {
            dim1,
            depth,
            rows,
            out_features: cols,
            num_params: params,
            data: vec![vec![vec![vec![vec![0.0; params]; cols]; rows]; depth]; dim1],
        }
    }

    pub fn slice_jacobian(&self, d: usize) -> Jacobian3D {
        let mut j3d = Jacobian3D::new(self.depth, self.rows, self.out_features, self.num_params);
        for z in 0..self.depth {
            for r in 0..self.rows {
                for c in 0..self.out_features {
                    for p in 0..self.num_params {
                        j3d.data[z][r][c][p] = self.data[d][z][r][c][p];
                    }
                }
            }
        }
        j3d
    }

    pub fn set_slice_jacobian(&mut self, d: usize, j3d: &Jacobian3D) {
        assert_eq!(j3d.depth, self.depth);
        assert_eq!(j3d.rows, self.rows);
        assert_eq!(j3d.out_features, self.out_features);
        assert_eq!(j3d.num_params, self.num_params);
        for z in 0..self.depth {
            for r in 0..self.rows {
                for c in 0..self.out_features {
                    for p in 0..self.num_params {
                        self.data[d][z][r][c][p] = j3d.data[z][r][c][p];
                    }
                }
            }
        }
    }
}





