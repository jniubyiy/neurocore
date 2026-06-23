// src/compute_manager/graph/forward/segments/connectors.rs

use crate::tensor::Tensor2D;
use crate::compute_manager::dim_change::DynamicTensor;
use crate::compute_manager::graph::model::MixedModel;
use crate::compute_manager::graph::types::DynamicContext;
use crate::layers::splitter_connector::SplitterConnector;
use crate::layers::combiner_connector::CombinerConnector;

impl MixedModel {
    pub(crate) fn process_splitter_connector_forward(
        &self,
        output_dims: Vec<usize>,
        batch_size: usize,
        streams: &mut Vec<Vec<DynamicTensor>>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
    ) {
        assert_eq!(streams.len(), 1);
        let num_outputs = output_dims.len();
        let input_stream = &streams[0];
        let mut data = Vec::with_capacity(batch_size);
        for d in input_stream.iter() {
            if let DynamicTensor::Dim1(t) = d {
                data.push(t.data[0].clone());
            } else {
                panic!("SplitterConnector поддерживает только Dim1 (Tensor2D)");
            }
        }
        let input_dim: usize = output_dims.iter().sum();
        let batch_tensor = DynamicTensor::Dim1(Tensor2D::new(data));

        let connector = SplitterConnector::new(input_dim, output_dims.clone());
        let (outputs, ctx) = connector.forward(&batch_tensor);

        let mut new_streams = vec![Vec::with_capacity(batch_size); num_outputs];
        for (out_idx, out_tensor) in outputs.iter().enumerate() {
            if let DynamicTensor::Dim1(t) = out_tensor {
                for r in 0..batch_size {
                    new_streams[out_idx].push(DynamicTensor::Dim1(Tensor2D::new(vec![t.data[r].clone()])));
                }
            } else {
                for _ in 0..batch_size {
                    new_streams[out_idx].push(DynamicTensor::Dim1(Tensor2D::zeros(1, output_dims[out_idx])));
                }
            }
        }

        for s in 0..batch_size {
            all_ctxs[s].push(ctx.clone());
        }

        *streams = new_streams;
    }

    pub(crate) fn process_combiner_connector_forward(
        &self,
        input_dims: Vec<usize>,
        batch_size: usize,
        streams: &mut Vec<Vec<DynamicTensor>>,
        all_ctxs: &mut Vec<Vec<DynamicContext>>,
    ) {
        assert_eq!(streams.len(), input_dims.len());
        let num_inputs = input_dims.len();
        let output_dim: usize = input_dims.iter().sum();

        let mut input_tensors = Vec::with_capacity(num_inputs);
        for s in 0..num_inputs {
            let stream = &streams[s];
            let mut data = Vec::with_capacity(batch_size);
            for d in stream.iter() {
                if let DynamicTensor::Dim1(t) = d {
                    data.push(t.data[0].clone());
                } else {
                    panic!("CombinerConnector поддерживает только Dim1 (Tensor2D)");
                }
            }
            input_tensors.push(DynamicTensor::Dim1(Tensor2D::new(data)));
        }

        let connector = CombinerConnector::new(input_dims.clone());
        let (output, ctx) = connector.forward(&input_tensors);

        let mut combined_stream = Vec::with_capacity(batch_size);
        if let DynamicTensor::Dim1(t) = &output {
            for r in 0..batch_size {
                combined_stream.push(DynamicTensor::Dim1(Tensor2D::new(vec![t.data[r].clone()])));
            }
        } else {
            for _ in 0..batch_size {
                combined_stream.push(DynamicTensor::Dim1(Tensor2D::zeros(1, output_dim)));
            }
        }

        for s in 0..batch_size {
            all_ctxs[s].push(ctx.clone());
        }

        *streams = vec![combined_stream];
    }
}