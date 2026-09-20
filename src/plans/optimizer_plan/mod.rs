// src/plans/optimizer_plan/mod.rs

pub mod cube;
pub mod chain;
pub mod expr;

pub use cube::OptimizerCube;
pub use chain::OptimizerChain;
pub use expr::OptimizerExpr;

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

    /// Превращает описание в готовую цепочку `OptimizerChain`.
    ///
    /// # Валидация (MIGRATION_PLAN.md §7, Фаза 1)
    ///
    /// В архитектуре v2 шаг оптимизатора разделён на две фазы
    /// (инвариант I-1):
    ///
    ///   * `modify_grads` — модификация градиента,
    ///   * `apply_update` — обновление параметров (`params -= grads`).
    ///
    /// Отсюда два требования к цепочке:
    ///
    ///   1. Кубик `ApplyUpdate` должен присутствовать **ровно один раз**.
    ///      Без него фаза `apply_update` не сможет обновить параметры.
    ///
    ///   2. Кубик `ApplyUpdate` должен быть **последним** в цепочке.
    ///      Он обновляет `params`, используя `grads`, — если после него
    ///      выполнить другой кубик, он будет работать со «старыми»
    ///      параметрами или «новыми» градиентами, что нарушает семантику
    ///      фазового порядка.
    ///
    /// Нарушение любого из этих требований — паника на этапе построения
    /// цепочки (fail-fast).
    pub fn build_chain(&self) -> OptimizerChain {
        // Валидация: ровно один ApplyUpdate, и он последний.
        let apply_count = self
            .cubes
            .iter()
            .filter(|c| matches!(c, OptCubeDesc::ApplyUpdate))
            .count();
        assert_eq!(
            apply_count, 1,
            "OptimizerDesc::build_chain: chain must contain exactly one \
             ApplyUpdate cube, found {}",
            apply_count
        );
        assert!(
            matches!(self.cubes.last(), Some(OptCubeDesc::ApplyUpdate)),
            "OptimizerDesc::build_chain: ApplyUpdate must be the last cube \
             in the chain (MIGRATION_PLAN.md §5.2, инвариант I-1)"
        );

        let mut chain = OptimizerChain::new();
        for cube in &self.cubes {
            match cube {
                OptCubeDesc::ScaleGradient(lr) => {
                    chain = chain.add(Box::new(
                        crate::optimizers::scale_gradient::ScaleGradient::new(*lr),
                    ));
                }
                OptCubeDesc::AddWeightDecay(decay) => {
                    chain = chain.add(Box::new(
                        crate::optimizers::add_weight_decay::AddWeightDecay::new(*decay),
                    ));
                }
                OptCubeDesc::Momentum(beta) => {
                    chain = chain.add(Box::new(
                        crate::optimizers::momentum::Momentum::new(*beta),
                    ));
                }
                OptCubeDesc::NesterovMomentum(beta) => {
                    chain = chain.add(Box::new(
                        crate::optimizers::nesterov_momentum::NesterovMomentum::new(*beta),
                    ));
                }
                OptCubeDesc::GradientClip { min, max } => {
                    chain = chain.add(Box::new(
                        crate::optimizers::gradient_clip::GradientClip::new(*min, *max),
                    ));
                }
                OptCubeDesc::Adam { beta1, beta2, eps } => {
                    chain = chain.add(Box::new(
                        crate::optimizers::adam::Adam::new(*beta1, *beta2, *eps),
                    ));
                }
                OptCubeDesc::ApplyUpdate => {
                    chain = chain.add(Box::new(
                        crate::optimizers::apply_update::ApplyUpdate::new(),
                    ));
                }
            }
        }
        chain
    }
}