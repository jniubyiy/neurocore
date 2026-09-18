// src/compute_manager/cpu/parallel/chunk_ops.rs
//
// Копирование чанков между общими буферами и per-chunk буферами.

use crate::compute_manager::operators_v2::memory_v2::buffer::{MatrixBufferHandle, TempMatrixPool};

/// Извлекает подматрицу строк `[start, end)` из `input` в новый буфер.
///
/// Возвращает новый буфер из пула, содержащий только строки
/// `[start, end)` (column-major).
pub(super) fn extract_chunk(
    input: &MatrixBufferHandle,
    start: usize,
    end: usize,
    pool: &mut TempMatrixPool,
) -> MatrixBufferHandle {
    let rows_total = input.rows();
    let cols = input.cols();
    assert!(end <= rows_total && start < end);

    let chunk_rows = end - start;
    let chunk = pool.acquire(chunk_rows, cols);

    let src_guard = input.read();
    let src = src_guard.as_slice().expect("CPU buffer");
    let mut dst_guard = chunk.write();
    let dst = dst_guard.as_slice_mut().expect("CPU buffer");

    for c in 0..cols {
        for r in 0..chunk_rows {
            dst[c * chunk_rows + r] = src[c * rows_total + start + r];
        }
    }
    chunk
}

/// Записывает чанк в выходной буфер, начиная со строки `start`.
pub(super) fn write_chunk(
    output: &MatrixBufferHandle,
    chunk: &MatrixBufferHandle,
    start: usize,
) {
    let out_rows = output.rows();
    let cols = output.cols();
    let chunk_rows = chunk.rows();

    if cols != chunk.cols() {
        eprintln!(
            "[WRITE-CHUNK-MISMATCH] output={}x{} chunk={}x{} start={} \
             (cols mismatch: output.cols()={} chunk.cols()={})",
            out_rows, cols, chunk_rows, chunk.cols(), start, cols, chunk.cols(),
        );
    }

    assert_eq!(cols, chunk.cols());
    assert!(start + chunk_rows <= out_rows);

    let mut out_guard = output.write();
    let out_slice = out_guard.as_slice_mut().expect("CPU buffer");
    let chunk_guard = chunk.read();
    let chunk_slice = chunk_guard.as_slice().expect("CPU buffer");

    for c in 0..cols {
        for r in 0..chunk_rows {
            out_slice[c * out_rows + start + r] = chunk_slice[c * chunk_rows + r];
        }
    }
}

/// Записывает чанк в диапазон `[out_start, out_end)` выходного буфера.
pub(super) fn write_chunk_to_range(
    output: &MatrixBufferHandle,
    chunk: &MatrixBufferHandle,
    out_start: usize,
    out_end: usize,
) {
    let chunk_rows = chunk.rows();
    assert_eq!(
        out_end - out_start,
        chunk_rows,
        "write_chunk_to_range: range size ({}) must match chunk rows ({})",
        out_end - out_start,
        chunk_rows,
    );
    write_chunk(output, chunk, out_start);
}