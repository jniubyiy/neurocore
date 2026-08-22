// src/layers/identity/identity.rs

use crate::layers::UniversalLayer;

pub struct Identity;

impl Identity {
    pub fn new() -> Self {
        Self
    }
}

impl UniversalLayer for Identity {
    fn as_identity(&self) -> Option<&Identity> {
        Some(self)
    }
}