// src/compute_manager/persistent_buffer.rs

use crate::compute_manager::device_spec::DeviceId;
use crate::compute_manager::graph::types::Segment;
use crate::compute_manager::memory_executor::{
    MemoryExecutor, MemoryDeviceKind, BufferPriority,
    executor::RawBufferId,
    ssd_cache::SsdHandle,
    TensorBufferId,
};
use crate::device_plan::plan::ComputeDevice;
use vulkano::memory::allocator::MemoryTypeFilter;

/// Идентификатор постоянного буфера, привязанного к конкретному устройству.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceBufferId {
    /// Буфер в оперативной памяти (CPU). Индекс используется для отслеживания в пуле MemoryExecutor.
    Cpu(usize),
    /// Буфер в видеопамяти GPU. RawBufferId из реестра сырых буферов.
    Gpu(RawBufferId),
    /// Буфер на SSD. Дескриптор SsdHandle.
    Ssd(SsdHandle),
}

/// Постоянный буфер, размещённый на определённом устройстве и живущий до явного освобождения.
#[derive(Clone)]
pub struct DeviceBuffer {
    pub id: DeviceBufferId,
    /// Устройство, на котором физически находится буфер.
    pub device: ComputeDevice,
    /// Количество элементов f32, которое вмещает буфер.
    pub size_elements: usize,
}

impl DeviceBuffer {
    /// Создать новый постоянный буфер в оперативной памяти (CPU).
    pub fn new_cpu(executor: &mut MemoryExecutor, elements: usize) -> Self {
        let kind = MemoryDeviceKind::HostRam;
        let tensor_id = executor
            .allocate_pinned(kind, elements, BufferPriority::High)
            .expect("Failed to allocate pinned CPU buffer");
        Self {
            id: DeviceBufferId::Cpu(tensor_id.0),
            device: ComputeDevice::Cpu { id: 0, threads: 0 },
            size_elements: elements,
        }
    }

    /// Создать новый постоянный буфер в видеопамяти GPU.
    pub fn new_gpu(executor: &mut MemoryExecutor, device_id: DeviceId, elements: usize) -> Self {
        let size_bytes = (elements * std::mem::size_of::<f32>()) as u64;
        let raw_id = executor.register_raw_buffer(
            device_id,
            size_bytes,
            MemoryTypeFilter::PREFER_DEVICE,
        );
        Self {
            id: DeviceBufferId::Gpu(raw_id),
            device: ComputeDevice::Gpu { id: device_id.0 },
            size_elements: elements,
        }
    }

    /// Освободить буфер и связанные с ним ресурсы.
    pub fn release(self, executor: &mut MemoryExecutor) {
        match self.id {
            DeviceBufferId::Cpu(tensor_id) => {
                executor
                    .deallocate_pinned(TensorBufferId(tensor_id))
                    .expect("Failed to deallocate pinned CPU buffer");
            }
            DeviceBufferId::Gpu(raw_id) => {
                executor.unregister_raw_buffer(raw_id);
            }
            DeviceBufferId::Ssd(_handle) => {
                // Удаление SSD-буфера пока не реализовано.
                // В будущем здесь будет вызов метода для освобождения SSD.
                // executor.deallocate_ssd(handle) – будет добавлено позже.
            }
        }
    }
}

/// Набор постоянных буферов, обслуживающих один вычислительный сегмент на всё время эпохи.
#[derive(Clone)]
pub struct SegmentPersistentBuffers {
    /// Входные буферы (по одному на каждый входной поток сегмента).
    pub inputs: Vec<DeviceBuffer>,
    /// Выходные буферы (по одному на каждый выходной поток).
    pub outputs: Vec<DeviceBuffer>,
    /// Буферы для сохранения контекста обратного прохода.
    /// Количество и размеры определяются типом сегмента.
    pub context: Vec<DeviceBuffer>,
    /// Флаг, указывающий, что все буферы выделены и готовы к использованию.
    pub allocated: bool,
}

impl SegmentPersistentBuffers {
    /// Создать полный набор буферов для заданного сегмента и целевого устройства.
    ///
    /// # Аргументы
    /// * `segment` – сегмент вычислительного графа.
    /// * `device` – вычислительное устройство, на котором будут располагаться буферы.
    /// * `batch_size` – максимальный размер батча (строки матриц).
    /// * `executor` – менеджер памяти, через который выделяются буферы.
    pub fn for_segment(
        segment: &Segment,
        device: &ComputeDevice,
        batch_size: usize,
        executor: &mut MemoryExecutor,
    ) -> Self {
        let (input_sizes, output_sizes, context_sizes) = estimate_buffer_sizes(segment, batch_size);

        let inputs: Vec<DeviceBuffer> = input_sizes
            .into_iter()
            .map(|sz| Self::create_buffer(device, sz, executor))
            .collect();

        let outputs: Vec<DeviceBuffer> = output_sizes
            .into_iter()
            .map(|sz| Self::create_buffer(device, sz, executor))
            .collect();

        let context: Vec<DeviceBuffer> = context_sizes
            .into_iter()
            .map(|sz| Self::create_buffer(device, sz, executor))
            .collect();

        Self {
            inputs,
            outputs,
            context,
            allocated: true,
        }
    }

    /// Освободить все буферы и вернуть ресурсы системе.
    pub fn release(self, executor: &mut MemoryExecutor) {
        for buf in self.inputs {
            buf.release(executor);
        }
        for buf in self.outputs {
            buf.release(executor);
        }
        for buf in self.context {
            buf.release(executor);
        }
    }

    fn create_buffer(device: &ComputeDevice, elements: usize, executor: &mut MemoryExecutor) -> DeviceBuffer {
        match device {
            ComputeDevice::Cpu { .. } => DeviceBuffer::new_cpu(executor, elements),
            ComputeDevice::Gpu { id } => DeviceBuffer::new_gpu(executor, DeviceId(*id), elements),
        }
    }
}

/// Оценивает необходимые размеры буферов (в количестве f32) для заданного сегмента и размера батча.
///
/// Возвращает кортеж:
/// * `Vec<usize>` – размеры входных буферов (по одному на каждый входной поток).
/// * `Vec<usize>` – размеры выходных буферов.
/// * `Vec<usize>` – размеры буферов контекста.
fn estimate_buffer_sizes(
    segment: &Segment,
    batch_size: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    match segment {
        Segment::UniversalProcessor(layers, _slices, stream_indices) => {
            let active_streams: Vec<usize> = match stream_indices {
                Some(indices) => indices.clone(),
                None => vec![0],
            };

            let first_layer = layers.first().expect("UniversalProcessor must have at least one layer");
            let in_features = first_layer.input_features();
            let input_sizes: Vec<usize> = active_streams.iter().map(|_| batch_size * in_features).collect();

            let last_layer = layers.last().unwrap();
            let out_features = last_layer.output_features();
            let output_sizes: Vec<usize> = active_streams.iter().map(|_| batch_size * out_features).collect();

            let mut context_sizes = Vec::new();
            for layer in layers.iter() {
                if layer.as_linear().is_some() {
                    context_sizes.push(batch_size * layer.input_features());
                } else if layer.as_relu().is_some() {
                    context_sizes.push(batch_size * layer.input_features());
                } else if layer.as_sigmoid().is_some() {
                    context_sizes.push(batch_size * layer.output_features());
                } else if layer.as_tanh().is_some() {
                    context_sizes.push(batch_size * layer.output_features());
                } else if layer.as_leaky_relu().is_some() {
                    context_sizes.push(batch_size * layer.input_features());
                } else if layer.as_softmax().is_some() {
                    context_sizes.push(batch_size * layer.output_features());
                } else if layer.as_dual_anchor().is_some()
                    || layer.as_soft_sparse_gate().is_some()
                    || layer.as_soft_keep_gate().is_some()
                {
                    context_sizes.push(batch_size * layer.input_features());
                } else if layer.as_memory().is_some() {
                    context_sizes.push(batch_size * layer.input_features());
                } else if layer.as_identity().is_some() {
                    context_sizes.push(batch_size * layer.input_features());
                } else if layer.as_unsqueeze().is_some() || layer.as_reduce_mean().is_some() {
                    // не должны попасть
                }
            }
            (input_sizes, output_sizes, context_sizes)
        }
        Segment::Splitter { input_dim, output_dims, .. } => {
            let p = output_dims[0];
            let q = output_dims[1];
            let input_sizes = vec![batch_size * input_dim];
            let output_sizes = vec![batch_size * p, batch_size * q];
            let context_sizes = vec![
                batch_size * input_dim,
                batch_size * p,
                batch_size * q,
            ];
            (input_sizes, output_sizes, context_sizes)
        }
        Segment::Combiner { input_dim, output_dim, .. } => {
            let input_sizes = vec![batch_size * input_dim, batch_size * input_dim];
            let output_sizes = vec![batch_size * output_dim];
            let context_sizes = vec![
                batch_size * input_dim,
                batch_size * input_dim,
                batch_size * output_dim,
            ];
            (input_sizes, output_sizes, context_sizes)
        }
        Segment::SplitterConnector { dim_a, dim_b } => {
            let input_sizes = vec![batch_size * dim_a, batch_size * dim_b];
            let output_sizes = input_sizes.clone();
            let context_sizes = vec![batch_size * dim_a];
            (input_sizes, output_sizes, context_sizes)
        }
        Segment::CombinerConnector { input_dims, .. } => {
            let input_sizes: Vec<usize> = input_dims.iter().map(|&d| batch_size * d).collect();
            let output_sizes = input_sizes.clone();
            let context_sizes = input_sizes.clone();
            (input_sizes, output_sizes, context_sizes)
        }
        Segment::Unsqueeze(target_dims) | Segment::ReduceMean(target_dims) => {
            let total = target_dims.iter().product::<usize>();
            let input_sizes = vec![batch_size * total];
            let output_sizes = input_sizes.clone();
            let context_sizes = input_sizes.clone();
            (input_sizes, output_sizes, context_sizes)
        }
    }
}