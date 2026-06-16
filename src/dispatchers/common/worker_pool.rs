use std::thread;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use std::sync::mpsc::{self, Sender};

type Task = Box<dyn FnOnce() + Send + 'static>;

pub struct WorkerPool {
    sender: Sender<Option<Task>>,
    workers: Vec<thread::JoinHandle<()>>,
    active_tasks: Arc<AtomicUsize>,
}

impl WorkerPool {
    pub fn new(num_threads: usize) -> Self {
        let (sender, receiver) = mpsc::channel::<Option<Task>>();
        let receiver = Arc::new(Mutex::new(receiver));
        let active_tasks = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let receiver = Arc::clone(&receiver);
            let active_tasks = Arc::clone(&active_tasks);
            let handle = thread::spawn(move || {
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
        WorkerPool { sender, workers, active_tasks }
    }

    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.active_tasks.fetch_add(1, Ordering::Release);
        self.sender.send(Some(Box::new(f))).expect("Worker pool has been dropped");
    }

    pub fn wait_all(&self) {
        while self.active_tasks.load(Ordering::Acquire) > 0 {
            std::hint::spin_loop();
        }
    }

    /// Возвращает количество потоков в пуле.
    pub fn num_workers(&self) -> usize {
        self.workers.len()
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