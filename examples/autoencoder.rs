// examples/autoencoder.rs

use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use neurocore::compute_manager::{Device, DynamicTensor};
use neurocore::model_plan::Plan;
use neurocore::tensor::Tensor2D;

mod models {
    use neurocore::model_plan::{Dim, LayerDesc, LayerKind};
    pub fn encoder() -> Vec<LayerDesc> {
        vec![
            LayerDesc::new("fc1", LayerKind::Linear, Dim::Dim1)
                .input(Dim::Dim1, &[4usize])
                .output(Dim::Dim1, &[2usize]),
            LayerDesc::new("sigm", LayerKind::Sigmoid, Dim::Dim1)
                .input(Dim::Dim1, &[2usize])
                .output(Dim::Dim1, &[2usize]),
        ]
    }
    pub fn decoder() -> Vec<LayerDesc> {
        vec![LayerDesc::new("fc2", LayerKind::Linear, Dim::Dim1)
            .input(Dim::Dim1, &[2usize])
            .output(Dim::Dim1, &[4usize])]
    }
}

mod losses {
    use neurocore::loss_plan::{Aggregation, ElementChain, LossDesc, Square, Sub};
    pub fn mse() -> LossDesc {
        let chain = ElementChain::new().add(Box::new(Sub)).add(Box::new(Square));
        LossDesc::from_chain(chain, Aggregation::Mean, 4, 1, 1)
    }
}

mod optimizers {
    use neurocore::optimizer_plan::{OptimizerDesc, OptCubeDesc};
    pub fn sgd_encoder() -> OptimizerDesc {
        OptimizerDesc::new().add(OptCubeDesc::ScaleGradient(0.01)).add(OptCubeDesc::ApplyUpdate)
    }
    pub fn sgd_decoder() -> OptimizerDesc {
        OptimizerDesc::new().add(OptCubeDesc::ScaleGradient(0.01)).add(OptCubeDesc::ApplyUpdate)
    }
}

fn run_training(
    device: Device,
    label: &str,
    initial_params_enc: Option<&[f32]>,
    initial_params_dec: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>, f32) {
    if matches!(device, Device::Gpu { .. }) {
        let device = device.clone();
        let label = label.to_string();
        let initial_params_enc = initial_params_enc.map(|p| p.to_vec());
        let initial_params_dec = initial_params_dec.map(|p| p.to_vec());
        let (tx, rx) = mpsc::channel();
        thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(move || {
                let (enc, dec, loss) = run_training_inner(
                    device,
                    &label,
                    initial_params_enc.as_deref(),
                    initial_params_dec.as_deref(),
                );
                tx.send((enc, dec, loss)).ok();
            })
            .unwrap();
        return rx.recv().unwrap();
    }
    run_training_inner(device, label, initial_params_enc, initial_params_dec)
}

fn run_training_inner(
    device: Device,
    label: &str,
    initial_params_enc: Option<&[f32]>,
    initial_params_dec: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>, f32) {
    println!("\n===== {} =====", label);
    let encoder_plan = Plan::from_layer_descs(models::encoder());
    let decoder_plan = Plan::from_layer_descs(models::decoder());
    let (mut encoder, mut decoder) = match (encoder_plan, decoder_plan) {
        (Ok(ep), Ok(dp)) => (ep.build_with_device(device.clone()), dp.build_with_device(device)),
        _ => {
            println!("[ERROR] Не удалось собрать модели автоэнкодера");
            return (vec![], vec![], 0.0);
        }
    };

    if let (Some(enc_p), Some(dec_p)) = (initial_params_enc, initial_params_dec) {
        encoder.param_store().lock().unwrap().set_all_params(enc_p);
        decoder.param_store().lock().unwrap().set_all_params(dec_p);
    } else {
        for model in [&mut encoder, &mut decoder] {
            let mut store = model.param_store().lock().unwrap();
            for i in 0..store.len() {
                store.set_param(i, rand::random::<f32>() * 0.01);
            }
        }
    }

    let x = Tensor2D::new(vec![vec![1.0, 2.0, 3.0, 4.0]]);
    let target = x.clone();
    let epochs = 500;

    let start = Instant::now();
    for epoch in 0..epochs {
        let (code, ctx_enc) = encoder.forward(DynamicTensor::Dim1(x.clone()));
        let (recon, ctx_dec) = decoder.forward(code);

        let (loss, delta_loss) = encoder.compute_loss(
            losses::mse(),
            &recon,
            &DynamicTensor::Dim1(target.clone()),
        );

        let (delta_code, grads_dec) = decoder.backward(&ctx_dec, delta_loss);
        decoder.update_params(optimizers::sgd_decoder(), &grads_dec[0]);

        let (_, grads_enc) = encoder.backward(&ctx_enc, delta_code);
        encoder.update_params(optimizers::sgd_encoder(), &grads_enc[0]);

        if epoch == 0 || epoch % 100 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss);
        }
    }
    let duration = start.elapsed();

    let (code, _) = encoder.forward(DynamicTensor::Dim1(x.clone()));
    let (final_recon, _) = decoder.forward(code);
    let (final_loss, _) = encoder.compute_loss(
        losses::mse(),
        &final_recon,
        &DynamicTensor::Dim1(target),
    );
    println!("Обучение завершено за {:?}", duration);
    println!("Финальный loss: {:.6}", final_loss);

    let enc_params = encoder.param_store().lock().unwrap().all_params_vec();
    let dec_params = decoder.param_store().lock().unwrap().all_params_vec();
    (enc_params, dec_params, final_loss)
}

fn main() {
    // Стандартные одиночные проходы
    run_training(Device::Cpu { threads: 1 }, "CPU (1 поток)", None, None);
    run_training(Device::Cpu { threads: 4 }, "CPU (4 потока)", None, None);
    run_training(Device::Gpu { id: 0 }, "GPU (id 0)", None, None);

    // CPU → GPU
    let (enc_cpu, dec_cpu, _) = run_training(Device::Cpu { threads: 1 }, "CPU (1 поток)", None, None);
    run_training(Device::Gpu { id: 0 }, "GPU после CPU", Some(&enc_cpu), Some(&dec_cpu));

    // GPU → CPU
    let (enc_gpu, dec_gpu, _) = run_training(Device::Gpu { id: 0 }, "GPU (id 0)", None, None);
    run_training(Device::Cpu { threads: 1 }, "CPU после GPU", Some(&enc_gpu), Some(&dec_gpu));
}

