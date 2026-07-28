// src/compute_manager/logger.rs

use std::cell::RefCell;
use std::io::{self, Write};
use std::sync::Mutex;

thread_local! {
    pub static LOG_BUF: RefCell<Vec<String>> = RefCell::new(Vec::new());
}

// Глобальный список адресов (для отладки не используется в хуке)
static ALL_BUFS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

pub fn register_thread() {
    LOG_BUF.with(|buf| {
        let ptr = buf as *const RefCell<Vec<String>> as usize;
        ALL_BUFS.lock().unwrap().push(ptr);
    });
}

pub fn log(msg: impl Into<String>) {
    LOG_BUF.with(|buf| {
        buf.borrow_mut().push(msg.into());
    });
}

pub fn dump_all_logs() {
    let mut stderr = io::stderr();
    let all_bufs = ALL_BUFS.lock().unwrap();
    for &ptr_usize in all_bufs.iter() {
        let buf_ptr = ptr_usize as *const RefCell<Vec<String>>;
        let buf = unsafe { &*buf_ptr };
        let msgs = buf.borrow();
        for msg in msgs.iter() {
            let _ = writeln!(stderr, "{}", msg);
        }
    }
    let _ = stderr.flush();
}

pub fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        // НЕ вызываем dump_all_logs, чтобы избежать deadlock
        // Вместо этого просто выводим информацию о панике и бэктрейс
        eprintln!("FATAL PANIC: {}", info);
        if let Some(location) = info.location() {
            eprintln!("  at {}:{}:{}", location.file(), location.line(), location.column());
        }
        let backtrace = std::backtrace::Backtrace::force_capture();
        eprintln!("{backtrace}");
        // Принудительно сбрасываем stderr
        let _ = io::stderr().flush();
        // Завершаем процесс
        std::process::abort();
    }));
}