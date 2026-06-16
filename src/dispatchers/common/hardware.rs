use once_cell::sync::Lazy;

#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub l3_cache_size: usize,
    pub frequency_mhz: u64,
}

impl CpuInfo {
    fn detect() -> Self {
        let logical = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        CpuInfo {
            logical_cores: logical,
            physical_cores: logical,
            l1_cache_size: 32 * 1024,
            l2_cache_size: 256 * 1024,
            l3_cache_size: 8 * 1024 * 1024,
            frequency_mhz: 3000,
        }
    }
}

pub static CPU_INFO: Lazy<CpuInfo> = Lazy::new(CpuInfo::detect);