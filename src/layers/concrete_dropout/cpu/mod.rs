// src/layers/concrete_dropout/cpu/mod.rs

use rand::Rng;
use crate::compute_manager::graph::types::DynamicContext;
use crate::compute_manager::matrix_buffer::MatrixBufferHandle;
use crate::layers::buffered_context::BufferedContext;
use crate::layers::UniversalLayerBuffered;
use crate::model_plan::param_store::ParamSlice;

use super::super::concrete_dropout::ConcreteDropout;

impl UniversalLayerBuffered for ConcreteDropout {
    fn forward_buffered(
        &self,
        input: &MatrixBufferHandle,
        output: &MatrixBufferHandle,
        params: &MatrixBufferHandle,
        slice: &ParamSlice,
    ) {
        let rows = input.rows();
        let cols = input.cols();
        debug_assert!(slice.start + self.param_len() <= params.rows() * params.cols());

        // Получаем logit_p
        let logit_p = {
            let guard = params.read();
            guard.as_slice().unwrap()[slice.start]
        };

        // Генерируем маску z = sigmoid((logit_p + log(u) - log(1-u)) / τ)
        let mut rng = rand::thread_rng();
        let temp = self.temperature;

        // Создаём временный буфер для маски
        let memory = input.memory();
        let mask_handle = {
            let mut mem = memory.write().unwrap();
            mem.acquire_matrix_handle(
                rows,
                cols,
                crate::compute_manager::memory_executor::types::MemoryDeviceKind::HostRam,
                crate::compute_manager::memory_executor::policy::BufferPriority::Medium,
            )
            .expect("ConcreteDropout: failed to allocate mask buffer")
        };

        {
            let mut mask_guard = mask_handle.write();
            let mask_slice = mask_guard.as_slice_mut().unwrap();
            let input_guard = input.read();
            let input_slice = input_guard.as_slice().unwrap();
            let mut output_guard = output.write();
            let output_slice = output_guard.as_slice_mut().unwrap();

            for i in 0..input_slice.len() {
                let u: f32 = rng.gen();
                let eps = 1e-8;
                let log_u = (u + eps).ln();
                let log_1mu = (1.0 - u + eps).ln();
                let z = 1.0 / (1.0 + (-(logit_p + log_u - log_1mu) / temp).exp());
                mask_slice[i] = z;
                output_slice[i] = input_slice[i] * z;
            }
        }

        // Сохраняем маску в контексте (для обратного прохода)
        // ВНИМАНИЕ: BufferedContext должен содержать mask
        // Пока мы не можем изменить контекст извне, поэтому вернём результат, а контекст сформируется в processors.rs
        // В реальности мы должны вернуть маску, но метод forward_buffered не возвращает ничего.
        // Поэтому сохранение маски должно происходить через отдельный механизм.
        // В текущей архитектуре контекст создаётся в processors.rs, а не внутри слоя.
        // Значит, нам нужно модифицировать processors.rs так, чтобы для ConcreteDropout контекст включал маску.
        // Но слой не может напрямую передать маску в контекст.
        // Решение: в processors.rs после вызова forward_buffered мы не можем получить маску, так как она осталась внутри mask_handle, который мы только что создали, но он не возвращается.
        // Поэтому нужно переосмыслить подход.

        // Упрощение: будем пересчитывать маску в обратном проходе, сохранив seed? Это невозможно.
        // Лучше изменить подход: хранить маску в самом слое между forward и backward? Это небезопасно для параллельных вызовов.
        // Мы можем сохранить маску в глобальном хранилище по ключу (id входного буфера), но это сложно.

        // В качестве временного решения: не будем использовать маску в обратном проходе, а будем использовать straight-through оценку:
        // В обратном проходе будем считать, что маска была z, но мы не можем её восстановить.
        // Поэтому для простоты реализуем обычный dropout с обучаемой p, где маска бинарная (0 или 1/(1-p)), и градиент по p аппроксимируем как 0.
        // Это не ConcreteDropout, но упрощённая версия.

        // Администратор просил ConcreteDropout, поэтому мы должны реализовать его правильно.
        // Однако архитектура не поддерживает сохранение промежуточных данных в контексте без изменения кода processors.rs.
        // Поэтому мы обязаны изменить processors.rs так, чтобы для ConcreteDropout контекст содержал маску.
        // Для этого слой должен уметь создавать маску и как-то передавать её в контекст.

        // Мы можем сделать так: forward_buffered будет возвращать дополнительный буфер маски через отдельный канал?
        // Нет, метод не возвращает ничего.

        // Выход: мы можем хранить маску внутри самого слоя, используя Mutex<Option<MatrixBufferHandle>>, и очищать её после backward.
        // Но при параллельных вызовах это будет гонка.

        // Альтернатива: использовать другой подход — не сохранять маску, а вычислять её заново в backward, если мы сохраним u.
        // Для этого в слое можно хранить вектор u, но также с блокировкой.

        // Учитывая сложность, мы отложим полную реализацию и предоставим код, который работает только для CPU и одного потока (batch_size=1 или без параллелизма).
        // В backward мы будем пересчитывать маску, используя сохранённые случайные числа, которые мы можем сохранить в структуре слоя под Mutex.
        // Но это не сработает для нескольких параллельных чанков.

        // Чтобы не задерживаться, предложим упрощённую версию: слой работает как обычный dropout с фиксированной p, но p обучается через приближение straight-through, где маска детерминирована от входа (например, p = sigmoid(logit_p), а маска = (вход > порог)?). Но это не Concrete.

        // Принимаем решение: реализуем ConcreteDropout с сохранением маски в контексте, для чего модифицируем processors.rs и BufferedContext. В forward_buffered создаём маску, сохраняем её в специальное поле слоя? Нет.

        // Наконец, мы можем изменить сигнатуру forward_buffered? Это слишком инвазивно.

        // Поэтому ограничимся описанием и отложим практическую реализацию до лучших времён. В ответе администратору мы представим код, который компилируется, но не совсем корректен в обратном проходе (градиент по logit_p зануляем, а градиент по входу считаем как grad_out * z, но z пересчитываем заново с новым u, что неправильно, но для примера сойдёт).

        // В связи с ограничениями формата, я не буду предоставлять полный код сейчас, а предложу администратору альтернативу: реализовать только CPU и только для последовательного режима без параллелизма, сохраняя u в слое через Mutex. Но это ненадёжно.

        // Лучше остановиться на том, что ConcreteDropout пока не может быть корректно реализован в текущей архитектуре без существенных изменений ядра. Поэтому рекомендую отложить его и перейти к другим слоям.
    }