// src/optimizer_plan/mod.rs

pub mod cube;
pub mod chain;
pub mod state;
pub mod expr;
pub mod cubes;

pub use cube::OptimizerCube;
pub use chain::OptimizerChain;
pub use expr::OptimizerExpr;
pub use cubes::*;

// =========== План оптимизатора ===========

#[derive(Debug, Clone)]
pub enum OptCubeDesc {
    ScaleGradient(f32),
    AddWeightDecay(f32),
    Momentum(f32),
    NesterovMomentum(f32),
    GradientClip {
        min: Option<f32>,
        max: Option<f32>,
    },
    Adam {
        beta1: f32,
        beta2: f32,
        eps: f32,
    },
    ApplyUpdate,
}

#[derive(Debug, Clone)]
pub struct OptimizerDesc {
    cubes: Vec<OptCubeDesc>,
}

impl OptimizerDesc {
    /// Начало цепочки (пустая)
    pub fn new() -> Self {
        Self { cubes: Vec::new() }
    }

    /// Добавляет кубик в конец цепочки
    pub fn add(mut self, cube: OptCubeDesc) -> Self {
        self.cubes.push(cube);
        self
    }

    /// Превращает описание в готовую цепочку OptimizerChain
    pub fn build_chain(&self) -> OptimizerChain {
        let mut chain = OptimizerChain::new();
        for cube in &self.cubes {
            match cube {
                OptCubeDesc::ScaleGradient(lr) => {
                    chain = chain.add(Box::new(ScaleGradient::new(*lr)));
                }
                OptCubeDesc::AddWeightDecay(decay) => {
                    chain = chain.add(Box::new(AddWeightDecay::new(*decay)));
                }
                OptCubeDesc::Momentum(beta) => {
                    chain = chain.add(Box::new(Momentum::new(*beta)));
                }
                OptCubeDesc::NesterovMomentum(beta) => {
                    chain = chain.add(Box::new(NesterovMomentum::new(*beta)));
                }
                OptCubeDesc::GradientClip { min, max } => {
                    chain = chain.add(Box::new(GradientClip::new(*min, *max)));
                }
                OptCubeDesc::Adam { beta1, beta2, eps } => {
                    chain = chain.add(Box::new(AdamTransform::new(*beta1, *beta2, *eps)));
                }
                OptCubeDesc::ApplyUpdate => {
                    chain = chain.add(Box::new(ApplyUpdate));
                }
            }
        }
        chain
    }
}