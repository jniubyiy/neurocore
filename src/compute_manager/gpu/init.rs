// src/compute_manager/gpu/init.rs

use std::sync::Arc;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo, QueueFlags,
};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::VulkanLibrary;

/// Информация о GPU, возвращаемая пользователю
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    /// Приблизительный объём видеопамяти (в мегабайтах)
    pub memory_mb: u64,
    pub device_type: PhysicalDeviceType,
}

/// Контекст Vulkan: экземпляр, логическое устройство и очередь вычислений
pub struct GpuContext {
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub queue: Arc<vulkano::device::Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub physical_device_index: usize,
}

/// Перечислить доступные GPU (физические устройства, поддерживающие compute).
/// Возвращает `None`, если Vulkan не загружен или произошла ошибка.
pub fn enumerate_gpus() -> Option<Vec<GpuInfo>> {
    let library = VulkanLibrary::new().ok()?;

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: InstanceExtensions::empty(),
            ..Default::default()
        },
    )
    .ok()?;

    let devices = instance.enumerate_physical_devices().ok()?;

    let mut gpu_list = Vec::new();
    for pd in devices {
        let props = pd.properties();
        // Суммируем размеры всех куч с флагом DEVICE_LOCAL
        let memory_mb = pd
            .memory_properties()
            .memory_heaps
            .iter()
            .filter(|h| {
                h.flags
                    .contains(vulkano::memory::MemoryHeapFlags::DEVICE_LOCAL)
            })
            .map(|h| h.size)
            .sum::<u64>()
            / (1024 * 1024);

        gpu_list.push(GpuInfo {
            name: props.device_name.clone(),
            memory_mb,
            device_type: props.device_type,
        });
    }

    Some(gpu_list)
}

/// Создать контекст GPU по индексу (начиная с 0) из списка, полученного `enumerate_gpus()`.
pub fn create_gpu_context(device_index: usize) -> Result<GpuContext, String> {
    let library = VulkanLibrary::new()
        .map_err(|e| format!("Не удалось загрузить Vulkan: {}", e))?;

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: InstanceExtensions::empty(),
            ..Default::default()
        },
    )
    .map_err(|e| format!("Ошибка создания Vulkan инстанса: {}", e))?;

    let physical_devices: Vec<_> = instance
        .enumerate_physical_devices()
        .map_err(|e| format!("Не удалось перечислить физические устройства: {}", e))?
        .collect();

    let physical = physical_devices
        .get(device_index)
        .ok_or_else(|| {
            format!(
                "GPU с индексом {} не найден. Доступно устройств: {}",
                device_index,
                physical_devices.len()
            )
        })?;

    // Ищем семейство очередей с поддержкой compute
    let queue_family_index = physical
        .queue_family_properties()
        .iter()
        .enumerate()
        .find(|(_, q)| q.queue_flags.intersects(QueueFlags::COMPUTE))
        .map(|(i, _)| i as u32)
        .ok_or_else(|| {
            format!(
                "Физическое устройство '{}' не поддерживает вычислительные операции",
                physical.properties().device_name
            )
        })?;

    // Создаём логическое устройство и одну compute-очередь
    let (device, mut queues) = Device::new(
        physical.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                queues: vec![1.0],
                ..Default::default()
            }],
            enabled_extensions: DeviceExtensions::empty(),
            ..Default::default()
        },
    )
    .map_err(|e| format!("Ошибка создания логического устройства: {}", e))?;

    let queue = queues
        .next()
        .ok_or("Не удалось получить очередь после создания устройства")?;

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    Ok(GpuContext {
        instance,
        device,
        queue,
        memory_allocator,
        physical_device_index: device_index,
    })
}