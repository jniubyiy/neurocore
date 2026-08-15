// src/compute_manager/memory_executor/data_mover.rs

use std::sync::Arc;

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{
    allocator::StandardCommandBufferAllocator,
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
};
use vulkano::sync::{self, GpuFuture};

use crate::compute_manager::gpu::init::GpuContext;

/// Синхронно копирует данные между двумя Vulkan-буферами.
/// Используется внутри `MemoryExecutor` и `GpuCompute`.
pub(crate) fn copy_buffer_sync(
    ctx: Arc<GpuContext>,
    src: Subbuffer<[f32]>,
    dst: Subbuffer<[f32]>,
) {
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        ctx.device.clone(),
        Default::default(),
    ));
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        ctx.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .copy_buffer(CopyBufferInfo::buffers(src, dst))
        .unwrap();
    let cb = builder.build().unwrap();
    let future = sync::now(ctx.device.clone())
        .then_execute(ctx.queue.clone(), cb)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();
}