use neurocore::tensor::Tensor2D;
use neurocore::jacobian::Jacobian2D;
use neurocore::layers::ReduceMean;
use neurocore::layers::Unsqueeze;
use neurocore::layers::{DimReduce, DimExpand};

fn main() {
    let params = 3;
    let x = Tensor2D::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let mut jx = Jacobian2D::new(2, 3, params);
    // заполним якобиан для примера
    for r in 0..2 {
        for c in 0..3 {
            for p in 0..params {
                jx.data[r][c][p] = 1.0;
            }
        }
    }

    let reducer = ReduceMean::new(0); // среднее по строкам
    let (y, jy) = reducer.reduce(&x, &jx);
    println!("Reduced (2D->1D): {:?}", y.data);
    println!("Jacobian:\n{:?}", jy.data);

    let expander = Unsqueeze::new(0); // добавит ось 0 размера 1
    let (x2, jx2) = expander.expand(&y, &jy);
    println!("Expanded back (1D->2D): {:?}", x2.data);
}




