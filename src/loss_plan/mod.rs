// src/loss_plan/mod.rs

// ==================== Элементарные кубики ====================
pub trait ElemCube: Send + Sync {
    fn in_size(&self) -> usize;
    fn out_size(&self) -> usize;
    fn forward_one(&self, input: &[f32]) -> Vec<f32>;
    fn backward_one(&self, input: &[f32], output_cache: &[f32], grad_out: &[f32]) -> Vec<f32>;
}

pub struct Sub;
impl ElemCube for Sub {
    fn in_size(&self) -> usize { 2 }
    fn out_size(&self) -> usize { 1 }
    fn forward_one(&self, input: &[f32]) -> Vec<f32> {
        vec![input[0] - input[1]]
    }
    fn backward_one(&self, _input: &[f32], _cache: &[f32], grad_out: &[f32]) -> Vec<f32> {
        vec![grad_out[0], -grad_out[0]]
    }
}

pub struct Square;
impl ElemCube for Square {
    fn in_size(&self) -> usize { 1 }
    fn out_size(&self) -> usize { 1 }
    fn forward_one(&self, input: &[f32]) -> Vec<f32> {
        let val = input[0];
        vec![val * val]
    }
    fn backward_one(&self, input: &[f32], _cache: &[f32], grad_out: &[f32]) -> Vec<f32> {
        vec![2.0 * input[0] * grad_out[0]]
    }
}

pub struct Log;
impl ElemCube for Log {
    fn in_size(&self) -> usize { 1 }
    fn out_size(&self) -> usize { 1 }
    fn forward_one(&self, input: &[f32]) -> Vec<f32> {
        vec![input[0].ln()]
    }
    fn backward_one(&self, input: &[f32], _cache: &[f32], grad_out: &[f32]) -> Vec<f32> {
        vec![grad_out[0] / input[0]]
    }
}

pub struct Neg;
impl ElemCube for Neg {
    fn in_size(&self) -> usize { 1 }
    fn out_size(&self) -> usize { 1 }
    fn forward_one(&self, input: &[f32]) -> Vec<f32> {
        vec![-input[0]]
    }
    fn backward_one(&self, _input: &[f32], _cache: &[f32], grad_out: &[f32]) -> Vec<f32> {
        vec![-grad_out[0]]
    }
}

pub struct Mul;
impl ElemCube for Mul {
    fn in_size(&self) -> usize { 2 }
    fn out_size(&self) -> usize { 1 }
    fn forward_one(&self, input: &[f32]) -> Vec<f32> {
        vec![input[0] * input[1]]
    }
    fn backward_one(&self, input: &[f32], _cache: &[f32], grad_out: &[f32]) -> Vec<f32> {
        vec![grad_out[0] * input[1], grad_out[0] * input[0]]
    }
}

// ==================== Новые кубики ====================

/// Модуль числа: |x|
pub struct Abs;
impl ElemCube for Abs {
    fn in_size(&self) -> usize { 1 }
    fn out_size(&self) -> usize { 1 }
    fn forward_one(&self, input: &[f32]) -> Vec<f32> {
        vec![input[0].abs()]
    }
    fn backward_one(&self, input: &[f32], _cache: &[f32], grad_out: &[f32]) -> Vec<f32> {
        let grad = if input[0] > 0.0 {
            grad_out[0]
        } else if input[0] < 0.0 {
            -grad_out[0]
        } else {
            0.0 // субградиент в нуле принят равным 0
        };
        vec![grad]
    }
}

/// Прибавление константы: x + scalar
pub struct AddScalar(pub f32);
impl ElemCube for AddScalar {
    fn in_size(&self) -> usize { 1 }
    fn out_size(&self) -> usize { 1 }
    fn forward_one(&self, input: &[f32]) -> Vec<f32> {
        vec![input[0] + self.0]
    }
    fn backward_one(&self, _input: &[f32], _cache: &[f32], grad_out: &[f32]) -> Vec<f32> {
        vec![grad_out[0]]
    }
}

/// Логарифм от (1 + x): ln(1 + x)
pub struct Log1p;
impl ElemCube for Log1p {
    fn in_size(&self) -> usize { 1 }
    fn out_size(&self) -> usize { 1 }
    fn forward_one(&self, input: &[f32]) -> Vec<f32> {
        let val = input[0] + 1.0;
        vec![val.ln()]
    }
    fn backward_one(&self, input: &[f32], _cache: &[f32], grad_out: &[f32]) -> Vec<f32> {
        vec![grad_out[0] / (1.0 + input[0])]
    }
}

/// Модуль разности двух чисел: |a - b|
pub struct AbsDiff;
impl ElemCube for AbsDiff {
    fn in_size(&self) -> usize { 2 }
    fn out_size(&self) -> usize { 1 }
    fn forward_one(&self, input: &[f32]) -> Vec<f32> {
        vec![(input[0] - input[1]).abs()]
    }
    fn backward_one(&self, input: &[f32], _cache: &[f32], grad_out: &[f32]) -> Vec<f32> {
        let diff = input[0] - input[1];
        let grad = if diff > 0.0 {
            grad_out[0]
        } else if diff < 0.0 {
            -grad_out[0]
        } else {
            0.0
        };
        vec![grad, -grad]
    }
}

// ==================== Кросс‑энтропия с логитами ====================
pub struct CrossEntropyWithLogits {
    pub num_classes: usize,
}

impl CrossEntropyWithLogits {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl ElemCube for CrossEntropyWithLogits {
    fn in_size(&self) -> usize { self.num_classes + 1 }
    fn out_size(&self) -> usize { 1 }

    fn forward_one(&self, input: &[f32]) -> Vec<f32> {
        let class = input[self.num_classes] as usize;
        let pred = &input[..self.num_classes];
        let max_val = pred.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = pred.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let loss = -exps[class].ln() + sum.ln();
        vec![loss]
    }

    fn backward_one(&self, input: &[f32], _cache: &[f32], grad_out: &[f32]) -> Vec<f32> {
        let class = input[self.num_classes] as usize;
        let pred = &input[..self.num_classes];
        let max_val = pred.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = pred.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let mut grad = vec![0.0; self.num_classes + 1];
        for j in 0..self.num_classes {
            grad[j] = exps[j] / sum;
        }
        grad[class] -= 1.0;
        for g in &mut grad {
            *g *= grad_out[0];
        }
        grad
    }
}

// ==================== Цепочка кубиков ====================
pub struct ElementChain {
    cubes: Vec<Box<dyn ElemCube>>,
}

impl ElementChain {
    pub fn new() -> Self {
        ElementChain { cubes: Vec::new() }
    }

    pub fn add(mut self, cube: Box<dyn ElemCube>) -> Self {
        self.cubes.push(cube);
        self
    }

    pub fn task_input_size(&self) -> usize {
        self.cubes.first().map(|c| c.in_size()).unwrap_or(0)
    }

    pub fn forward_one_full(&self, input: &[f32]) -> (Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>) {
        let mut intermediates = Vec::new();
        let mut current = input.to_vec();
        for cube in &self.cubes {
            let out = cube.forward_one(&current);
            intermediates.push((current.clone(), out.clone()));
            current = out;
        }
        (current, intermediates)
    }
}

// ==================== Агрегация ====================
pub enum Aggregation {
    Sum,
    Mean,
}

// ==================== Выражение потерь ====================
pub struct LossExpr {
    chain: ElementChain,
    aggregation: Aggregation,
    total_tasks: usize,
    pred_features: usize,
    target_features: usize,
}

impl LossExpr {
    pub fn new(
        chain: ElementChain,
        aggregation: Aggregation,
        total_tasks: usize,
        pred_features: usize,
        target_features: usize,
    ) -> Self {
        Self {
            chain,
            aggregation,
            total_tasks,
            pred_features,
            target_features,
        }
    }

    pub fn num_tasks(&self) -> usize {
        self.total_tasks
    }

    pub fn task_input_size(&self) -> usize {
        self.chain.task_input_size()
    }

    pub fn pred_features(&self) -> usize {
        self.pred_features
    }

    pub fn target_features(&self) -> usize {
        self.target_features
    }

    pub fn forward_chunk(&self, _start: usize, count: usize, chunk_inputs: &[f32], out_loss: &mut [f32]) {
        let in_size = self.chain.task_input_size();
        for i in 0..count {
            let input = &chunk_inputs[i * in_size..(i + 1) * in_size];
            let (final_out, _) = self.chain.forward_one_full(input);
            out_loss[i] = final_out[0];
        }
    }

    pub fn backward_chunk(
        &self,
        _start: usize,
        count: usize,
        chunk_inputs: &[f32],
        _out_loss: &[f32],
        grad_loss: &[f32],
        grad_pred: &mut [f32],
    ) {
        let in_size = self.chain.task_input_size();
        for i in 0..count {
            let input = &chunk_inputs[i * in_size..(i + 1) * in_size];
            let (_, intermediates) = self.chain.forward_one_full(input);
            let mut grad = vec![grad_loss[i]];
            for (idx, cube) in self.chain.cubes.iter().enumerate().rev() {
                let (inp, outp) = &intermediates[idx];
                grad = cube.backward_one(inp, outp, &grad);
            }
            grad_pred[i * in_size..(i + 1) * in_size].copy_from_slice(&grad);
        }
    }

    pub fn aggregate_loss(&self, loss_parts: &[f32]) -> f32 {
        let sum: f32 = loss_parts.iter().sum();
        let n = self.total_tasks as f32;
        match self.aggregation {
            Aggregation::Sum => sum,
            Aggregation::Mean => sum / n,
        }
    }

    pub fn aggregate_grad(&self, grad_parts: &[f32]) -> Vec<f32> {
        let n = self.total_tasks as f32;
        match self.aggregation {
            Aggregation::Sum => grad_parts.to_vec(),
            Aggregation::Mean => grad_parts.iter().map(|g| g / n).collect(),
        }
    }
}

// ==================== План потерь ====================
pub struct LossDesc {
    pub chain: ElementChain,
    pub aggregation: Aggregation,
    pub total_tasks: usize,
    pub pred_features: usize,
    pub target_features: usize,
}

impl LossDesc {
    pub fn from_chain(
        chain: ElementChain,
        aggregation: Aggregation,
        total_tasks: usize,
        pred_features: usize,
        target_features: usize,
    ) -> Self {
        Self {
            chain,
            aggregation,
            total_tasks,
            pred_features,
            target_features,
        }
    }

    pub fn build(self) -> std::sync::Arc<LossExpr> {
        std::sync::Arc::new(LossExpr::new(
            self.chain,
            self.aggregation,
            self.total_tasks,
            self.pred_features,
            self.target_features,
        ))
    }
}