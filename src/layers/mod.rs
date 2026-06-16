pub mod layers1d;
pub mod layers2d;
pub mod layers3d;
pub mod layers4d;
pub mod layers5d;
pub mod layers_special;

// 1D
pub use layers1d::{
    Layer, LayerInfo,
    LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer, TanhLayer, MemoryLayer,
    Sequential,
    LayerBuilder, LinearLayerBuilder, ReLULayerBuilder, SigmoidLayerBuilder, SoftmaxLayerBuilder,
};

// 2D
pub use layers2d::{
    Layer2D,
    Linear2D, ReLU2D, Sigmoid2D, Softmax2D, Tanh2D, Memory2D,
};

// 3D
pub use layers3d::{
    Layer3D,
    Linear3D, ReLU3D, Sigmoid3D, Softmax3D, Tanh3D, Memory3D,
};

// 4D
pub use layers4d::{
    Layer4D,
    Linear4D, ReLU4D, Sigmoid4D, Softmax4D, Tanh4D, Memory4D,
};

// 5D
pub use layers5d::{
    Layer5D,
    Linear5D, ReLU5D, Sigmoid5D, Softmax5D, Tanh5D, Memory5D,
};

// Специальные
pub use layers_special::{
    DimReduce, DimExpand,
    ReduceMean, Unsqueeze,
};


