pub mod reduce_dim;
pub mod expand_dim;

pub use reduce_dim::ReduceMean;
pub use expand_dim::Unsqueeze;

pub trait DimReduce<InputTensor, InputJacobian, OutputTensor, OutputJacobian> {
    fn reduce(&self, input: &InputTensor, j_input: &InputJacobian) -> (OutputTensor, OutputJacobian);
    fn param_count(&self) -> usize;
    fn update_params(&mut self, lr: f32, grad: &[f32]);
    fn get_params(&self) -> Vec<f32>;
    fn set_params(&mut self, values: &[f32]);
}

pub trait DimExpand<InputTensor, InputJacobian, OutputTensor, OutputJacobian> {
    fn expand(&self, input: &InputTensor, j_input: &InputJacobian) -> (OutputTensor, OutputJacobian);
    fn param_count(&self) -> usize;
    fn update_params(&mut self, lr: f32, grad: &[f32]);
    fn get_params(&self) -> Vec<f32>;
    fn set_params(&mut self, values: &[f32]);
}





