# Shader Restrictions for NeuroCore

All compute shaders in this project MUST comply with the following rules to
ensure compatibility with the broadest range of Vulkan devices (including
integrated GPUs).

## ❌ Forbidden Extensions
Do NOT enable or use any of these extensions in any `.comp` file:
- `GL_EXT_shader_atomic_float` (or `GL_EXT_shader_atomic_float2`)
- `GL_EXT_shader_atomic_float_min_max`
- Any other extension not part of core Vulkan 1.0 / SPIR-V 1.0

## ❌ Forbidden Capabilities
Do NOT rely on the following SPIR-V capabilities:
- `AtomicFloat32MinMaxEXT`
- `AtomicFloat16MinMaxEXT`

## ✅ Allowed Functionality
Use only core GLSL 4.50 / Vulkan 1.0 features, including:
- Standard atomic operations on **integers** (`atomicMin`, `atomicMax`, `atomicAdd`, etc.)
- To emulate float atomics, convert bits via `floatBitsToInt` / `floatBitsToUint`
  and operate on integer memory. Example:
  ```glsl
  int iv = floatBitsToInt(myFloat);
  atomicMin(intBuffer[index], iv);
  // read back: intBitsToFloat(intBuffer[index])
  Shared memory (workgroup) operations are allowed.
  Push constants, storage buffers, uniform variables are all fine.