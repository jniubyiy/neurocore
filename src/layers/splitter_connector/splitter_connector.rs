// src/layers/splitter_connector/splitter_connector.rs

use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::types::DynamicContext;

pub struct SplitterConnector {
    input_dim: usize,
    output_dims: Vec<usize>,
}

impl SplitterConnector {
    pub fn new(input_dim: usize, output_dims: Vec<usize>) -> Self {
        let sum: usize = output_dims.iter().sum();
        assert_eq!(input_dim, sum, "Сумма выходных размерностей должна равняться входной");
        Self { input_dim, output_dims }
    }

    pub fn input_dim(&self) -> usize { self.input_dim }
    pub fn output_dims(&self) -> &[usize] { &self.output_dims }

    pub fn forward(
        &self,
        input: &DynamicTensor,
    ) -> (Vec<DynamicTensor>, DynamicContext) {
        match input {
            DynamicTensor::Dim1(t) => {
                let batch = t.dim1;
                let features = t.dim2;
                assert_eq!(features, self.input_dim);
                let mut outputs = Vec::with_capacity(self.output_dims.len());
                let mut offset = 0;
                for &dim in &self.output_dims {
                    let mut data = Vec::with_capacity(batch);
                    for r in 0..batch {
                        data.push(t.data[r][offset..offset + dim].to_vec());
                    }
                    outputs.push(DynamicTensor::Dim1(crate::tensor::Tensor2D::new(data)));
                    offset += dim;
                }
                let ctx = DynamicContext::Ctx1D(
                    crate::layers::context1d::LayerContext1D::SplitterConnector {
                        input: t.clone(),
                    },
                );
                (outputs, ctx)
            }
            DynamicTensor::Dim2(t) => {
                let batch = t.dim1;
                let dim2 = t.dim2;
                let features = t.dim3;
                assert_eq!(features, self.input_dim);
                let mut outputs = Vec::with_capacity(self.output_dims.len());
                let mut offset = 0;
                for &dim in &self.output_dims {
                    let mut vol = Vec::with_capacity(batch);
                    for i in 0..batch {
                        let mut plane = Vec::with_capacity(dim2);
                        for j in 0..dim2 {
                            plane.push(t.data[i][j][offset..offset + dim].to_vec());
                        }
                        vol.push(plane);
                    }
                    outputs.push(DynamicTensor::Dim2(crate::tensor::Tensor3D::new(vol)));
                    offset += dim;
                }
                let ctx = DynamicContext::Ctx2D(
                    crate::layers::context2d::LayerContext::SplitterConnector {
                        input: t.clone(),
                    },
                );
                (outputs, ctx)
            }
            DynamicTensor::Dim3(t) => {
                let batch = t.dim1;
                let dim2 = t.dim2;
                let dim3 = t.dim3;
                let features = t.dim4;
                assert_eq!(features, self.input_dim);
                let mut outputs = Vec::with_capacity(self.output_dims.len());
                let mut offset = 0;
                for &dim in &self.output_dims {
                    let mut hyper = Vec::with_capacity(batch);
                    for i in 0..batch {
                        let mut vol = Vec::with_capacity(dim2);
                        for j in 0..dim2 {
                            let mut plane = Vec::with_capacity(dim3);
                            for k in 0..dim3 {
                                plane.push(t.data[i][j][k][offset..offset + dim].to_vec());
                            }
                            vol.push(plane);
                        }
                        hyper.push(vol);
                    }
                    outputs.push(DynamicTensor::Dim3(crate::tensor::Tensor4D::new(hyper)));
                    offset += dim;
                }
                let ctx = DynamicContext::Ctx3D(
                    crate::layers::context3d::LayerContext3D::SplitterConnector {
                        input: t.clone(),
                    },
                );
                (outputs, ctx)
            }
            DynamicTensor::Dim4(t) => {
                let batch = t.dim1;
                let dim2 = t.dim2;
                let dim3 = t.dim3;
                let dim4 = t.dim4;
                let features = t.dim5;
                assert_eq!(features, self.input_dim);
                let mut outputs = Vec::with_capacity(self.output_dims.len());
                let mut offset = 0;
                for &dim in &self.output_dims {
                    let mut mega = Vec::with_capacity(batch);
                    for i in 0..batch {
                        let mut hyper = Vec::with_capacity(dim2);
                        for j in 0..dim2 {
                            let mut vol = Vec::with_capacity(dim3);
                            for k in 0..dim3 {
                                let mut plane = Vec::with_capacity(dim4);
                                for l in 0..dim4 {
                                    plane.push(t.data[i][j][k][l][offset..offset + dim].to_vec());
                                }
                                vol.push(plane);
                            }
                            hyper.push(vol);
                        }
                        mega.push(hyper);
                    }
                    outputs.push(DynamicTensor::Dim4(crate::tensor::Tensor5D::new(mega)));
                    offset += dim;
                }
                let ctx = DynamicContext::Ctx4D(
                    crate::layers::context4d::LayerContext4D::SplitterConnector {
                        input: t.clone(),
                    },
                );
                (outputs, ctx)
            }
        }
    }

    pub fn backward(
        &self,
        ctx: &DynamicContext,
        deltas: &[DynamicTensor],
    ) -> (DynamicTensor, Vec<f32>) {
        match ctx {
            DynamicContext::Ctx1D(c) => {
                let input = match c {
                    crate::layers::context1d::LayerContext1D::SplitterConnector { input } => input,
                    _ => panic!("Expected SplitterConnector context"),
                };
                let batch = input.dim1;
                let mut combined = vec![vec![0.0; self.input_dim]; batch];
                let mut offset = 0;
                for (idx, delta) in deltas.iter().enumerate() {
                    if let DynamicTensor::Dim1(d) = delta {
                        for r in 0..batch {
                            combined[r][offset..offset + self.output_dims[idx]]
                                .copy_from_slice(&d.data[r]);
                        }
                    } else {
                        panic!("Mismatched delta dimension in SplitterConnector backward");
                    }
                    offset += self.output_dims[idx];
                }
                (DynamicTensor::Dim1(crate::tensor::Tensor2D::new(combined)), vec![])
            }
            DynamicContext::Ctx2D(c) => {
                let input = match c {
                    crate::layers::context2d::LayerContext::SplitterConnector { input } => input,
                    _ => panic!("Expected SplitterConnector context"),
                };
                let batch = input.dim1;
                let dim2 = input.dim2;
                let mut combined = vec![vec![vec![0.0; self.input_dim]; dim2]; batch];
                let mut offset = 0;
                for (idx, delta) in deltas.iter().enumerate() {
                    if let DynamicTensor::Dim2(d) = delta {
                        for i in 0..batch {
                            for j in 0..dim2 {
                                combined[i][j][offset..offset + self.output_dims[idx]]
                                    .copy_from_slice(&d.data[i][j]);
                            }
                        }
                    } else {
                        panic!("Mismatched delta dimension");
                    }
                    offset += self.output_dims[idx];
                }
                (DynamicTensor::Dim2(crate::tensor::Tensor3D::new(combined)), vec![])
            }
            DynamicContext::Ctx3D(c) => {
                let input = match c {
                    crate::layers::context3d::LayerContext3D::SplitterConnector { input } => input,
                    _ => panic!("Expected SplitterConnector context"),
                };
                let batch = input.dim1;
                let dim2 = input.dim2;
                let dim3 = input.dim3;
                let mut combined = vec![vec![vec![vec![0.0; self.input_dim]; dim3]; dim2]; batch];
                let mut offset = 0;
                for (idx, delta) in deltas.iter().enumerate() {
                    if let DynamicTensor::Dim3(d) = delta {
                        for i in 0..batch {
                            for j in 0..dim2 {
                                for k in 0..dim3 {
                                    combined[i][j][k][offset..offset + self.output_dims[idx]]
                                        .copy_from_slice(&d.data[i][j][k]);
                                }
                            }
                        }
                    } else {
                        panic!("Mismatched delta dimension");
                    }
                    offset += self.output_dims[idx];
                }
                (DynamicTensor::Dim3(crate::tensor::Tensor4D::new(combined)), vec![])
            }
            DynamicContext::Ctx4D(c) => {
                let input = match c {
                    crate::layers::context4d::LayerContext4D::SplitterConnector { input } => input,
                    _ => panic!("Expected SplitterConnector context"),
                };
                let batch = input.dim1;
                let dim2 = input.dim2;
                let dim3 = input.dim3;
                let dim4 = input.dim4;
                let mut combined = vec![vec![vec![vec![vec![0.0; self.input_dim]; dim4]; dim3]; dim2]; batch];
                let mut offset = 0;
                for (idx, delta) in deltas.iter().enumerate() {
                    if let DynamicTensor::Dim4(d) = delta {
                        for i in 0..batch {
                            for j in 0..dim2 {
                                for k in 0..dim3 {
                                    for l in 0..dim4 {
                                        combined[i][j][k][l][offset..offset + self.output_dims[idx]]
                                            .copy_from_slice(&d.data[i][j][k][l]);
                                    }
                                }
                            }
                        }
                    } else {
                        panic!("Mismatched delta dimension");
                    }
                    offset += self.output_dims[idx];
                }
                (DynamicTensor::Dim4(crate::tensor::Tensor5D::new(combined)), vec![])
            }
        }
    }

    pub fn param_len(&self) -> usize { 0 }
}