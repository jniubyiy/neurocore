// src/compute_manager/cpu/worker_pool.rs

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

type Task = Box<dyn FnOnce() + Send + 'static>;

thread_local! {
    static WORKER_INDEX: Cell<usize> = Cell::new(0);
}

/// Пул рабочих потоков с возможностью получить индекс текущего потока.
pub struct WorkerPool {
    sender: Sender<Option<Task>>,
    workers: Vec<thread::JoinHandle<()>>,
    active_tasks: Arc<AtomicUsize>,
}

impl WorkerPool {
    /// Создаёт пул с заданным числом потоков.
    pub fn new(num_threads: usize) -> Self {
        let (sender, receiver) = mpsc::channel::<Option<Task>>();
        let receiver = Arc::new(Mutex::new(receiver));
        let active_tasks = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::with_capacity(num_threads);

        for worker_id in 0..num_threads {
            let receiver = Arc::clone(&receiver);
            let active_tasks = Arc::clone(&active_tasks);
            let handle = thread::spawn(move || {
                // Устанавливаем индекс текущего потока
                WORKER_INDEX.set(worker_id);

                loop {
                    let task = {
                        let rx = receiver.lock().unwrap();
                        rx.recv().unwrap()
                    };
                    match task {
                        Some(task) => {
                            task();
                            active_tasks.fetch_sub(1, Ordering::Release);
                        }
                        None => break,
                    }
                }
            });
            workers.push(handle);
        }

        WorkerPool {
            sender,
            workers,
            active_tasks,
        }
    }

    /// Отправляет задачу на выполнение одному из рабочих потоков.
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.active_tasks.fetch_add(1, Ordering::Release);
        self.sender
            .send(Some(Box::new(f)))
            .expect("Worker pool has been dropped");
    }

    /// Ожидает завершения всех активных задач.
    pub fn wait_all(&self) {
        while self.active_tasks.load(Ordering::Acquire) > 0 {
            std::hint::spin_loop();
        }
    }

    /// Возвращает количество рабочих потоков в пуле.
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Возвращает индекс текущего рабочего потока (0..num_workers-1).
    /// Может быть вызван только внутри задачи, выполняемой в пуле.
    /// Вне пула возвращает 0 (но это не должно использоваться).
    pub fn current_worker_index() -> usize {
        WORKER_INDEX.with(|cell| cell.get())
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        for _ in &self.workers {
            self.sender.send(None).ok();
        }
        for handle in self.workers.drain(..) {
            handle.join().ok();
        }
    }
}