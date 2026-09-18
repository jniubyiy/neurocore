// src/compute_manager/cpu/send_ptr.rs

pub struct SendPtr<T>(pub *mut T);

unsafe impl<T> Send for SendPtr<T> {}

impl<T> SendPtr<T> {
    pub unsafe fn as_mut(&self) -> &mut T {
        &mut *self.0
    }
}