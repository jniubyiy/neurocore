use neurocore::tensor::{Tensor2D, Tensor1D};
use neurocore::layers::{ReduceMean, Unsqueeze};
use neurocore::layers::layers_special::{DimReduce, DimExpand};

fn main() {
    let x = Tensor2D::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let reducer = ReduceMean::new(0); // среднее по строкам
    let y: Tensor1D = reducer.reduce(&x);
    println!("Reduced (2D->1D): {:?}", y.data);

    let expander = Unsqueeze::new(0);
    let x2: Tensor2D = expander.expand(&y);
    println!("Expanded back (1D->2D): {:?}", x2.data);
}




