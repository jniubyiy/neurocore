// examples/autoencoder.rs
// Автоэнкодер 4 -> 2 -> 4 (Dim1) с 7 вариантами обучения через run_training!.

use neurocore::tensor::Tensor2D;
use neurocore::training_plan::ProfileMode;

mod models {
    use neurocore::model_plan::{LayerKind, LayerDesc};
    use neurocore::shape;
    pub fn autoencoder() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[4]))
                .output(shape!(batch, A[2])),
            LayerDesc::new(LayerKind::Sigmoid)
                .input(shape!(batch, A[2]))
                .output(shape!(batch, A[2])),
            LayerDesc::new(LayerKind::Linear)
                .input(shape!(batch, A[2]))
                .output(shape!(batch, A[4])),
        ]
    }
}

mod losses {
    use neurocore::loss_plan::{
        Aggregation, ElementChain, LossDesc, Square, Sub, SumColumns,
    };
    pub fn mse() -> LossDesc {
        let chain = ElementChain::new()
            .add(Box::new(Sub::new(4)))
            .add(Box::new(Square))
            .add(Box::new(SumColumns));
        LossDesc::from_chain(chain, Aggregation::Mean, 1, 4, 4)
    }
}

mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};
    pub fn sgd() -> OptimizerDesc {
        OptimizerDesc::new()
            .add(OptCubeDesc::ScaleGradient(0.01))
            .add(OptCubeDesc::ApplyUpdate)
    }
}

fn data() -> Tensor2D {
    Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]])
}

fn base_training() -> neurocore::training_plan::TrainingPlan {
    use neurocore::training_plan::plan::{TrainingPlan, DataSource, Initializer};
    TrainingPlan::new()
        .model(models::autoencoder)
        .loss(losses::mse())
        .optimizer(optimizers::sgd())
        .epochs(100)
        .batch_size(1)
        .train_data(DataSource::from_tensor2d(data()))
        .target_data(DataSource::from_tensor2d(data()))
        .init_weights(Initializer::RandomUniform { min: -0.1, max: 0.1 })
        .seed(42)
        .output_tensors(vec!["prediction".to_string()])
}

fn profiled_training() -> neurocore::training_plan::TrainingPlan {
    base_training().profile(ProfileMode::Full)
}

macro_rules! device_plan_v {
    ($name:ident, $cpu:expr, $ram:expr, $gpu:expr, $vram:expr, $ssd:expr) => {
        mod $name {
            use neurocore::device_plan::DevicePlan;
            pub fn plan() -> DevicePlan {
                let p = DevicePlan::empty().cpu(0, $cpu).ram(0, $ram);
                let p = if $gpu { p.gpu(0).vram(0, 0, $vram) } else { p };
                if $ssd { p.ssd(0, "neurocore_ssd_cache", 5000) } else { p }
            }
        }
    };
}

device_plan_v!(device_plan_v1, 1, 8192, false, 0, false);
device_plan_v!(device_plan_v2, 4, 8192, false, 0, false);
device_plan_v!(device_plan_v3, 2, 8192, true, 4096, false);
device_plan_v!(device_plan_v4_cpu, 1, 8192, false, 0, false);
device_plan_v!(device_plan_v4_gpu, 2, 8192, true, 4096, false);
device_plan_v!(device_plan_v5_gpu, 2, 8192, true, 4096, false);
device_plan_v!(device_plan_v5_cpu, 1, 8192, false, 0, false);
device_plan_v!(device_plan_v6, 4, 8192, false, 0, true);
device_plan_v!(device_plan_v7, 4, 8192, true, 4096, false);

fn print_result(label: &str, r: &neurocore::training_plan::execution::TrainingResult) {
    println!(
        "{}  time={:.3}s | best_loss={:.6} @ epoch {} | zero_loss_epoch={:?}",
        label, r.training_time_secs, r.best_loss, r.best_epoch, r.zero_loss_epoch
    );
    if let Some(ref profile) = r.profile {
        println!("{}", profile.report());
    }
}

fn main() {
    let r1 = neurocore::run_training!(base_training, device = device_plan_v1::plan);
    print_result("V1 CPU1", &r1);
    let r2 = neurocore::run_training!(base_training, device = device_plan_v2::plan);
    print_result("V2 CPU4", &r2);
    let r3 = neurocore::run_training!(base_training, device = device_plan_v3::plan);
    print_result("V3 GPU ", &r3);
    let r4a = neurocore::run_training!(base_training, device = device_plan_v4_cpu::plan);
    print_result("V4a CPU", &r4a);
    let r4b = neurocore::run_training!(base_training, device = device_plan_v4_gpu::plan);
    print_result("V4b GPU", &r4b);
    let r5a = neurocore::run_training!(base_training, device = device_plan_v5_gpu::plan);
    print_result("V5a GPU", &r5a);
    let r5b = neurocore::run_training!(base_training, device = device_plan_v5_cpu::plan);
    print_result("V5b CPU", &r5b);
    let r6 = neurocore::run_training!(base_training, device = device_plan_v6::plan);
    print_result("V6 SSD", &r6);
    let r7 = neurocore::run_training!(profiled_training, device = device_plan_v7::plan);
    print_result("V7 Prof", &r7);
}

