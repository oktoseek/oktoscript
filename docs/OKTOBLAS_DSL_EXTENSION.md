# OktoScript × OktoBLAS - DSL Extension Proposal

**Version:** 1.3 (Draft)  
**Status:** Proposed for OktoScript v1.3  
**Author:** OktoSeek AI

---

## Overview

This document proposes new OktoScript blocks and commands that directly leverage OktoBLAS capabilities, giving users fine-grained control over GPU acceleration and BLAS operations.

---

## 🆕 BLAS Block (Proposed for v1.3)

The `BLAS` block allows users to configure OktoBLAS-specific settings for training and inference.

```ebnf
<blas_block> ::=
  "BLAS" "{"
      [<blas_backend>]
      [<blas_precision>]
      [<blas_kernel>]
      [<blas_autoselect>]
      [<blas_benchmark_mode>]
      [<blas_stream_count>]
      [<blas_memory_pool>]
  "}"

<blas_backend> ::=
  "backend" ":" ("oktoblas" | "cublas" | "auto")

<blas_precision> ::=
  "precision" ":" ("fp16" | "fp32" | "tf32" | "bf16" | "auto")

<blas_kernel> ::=
  "kernel" ":" ("final_v1" | "best_v3" | "pure" | "flash_attention" | "auto")

<blas_autoselect> ::=
  "autoselect" ":" <boolean>

<blas_benchmark_mode> ::=
  "benchmark_mode" ":" <boolean>

<blas_stream_count> ::=
  "streams" ":" <number>

<blas_memory_pool> ::=
  "memory_pool" ":" ("enabled" | "disabled" | "auto")
```

### Example

```okt
BLAS {
    backend: "oktoblas"
    precision: "fp16"
    kernel: "final_v1"
    autoselect: true
    streams: 4
    memory_pool: "enabled"
}
```

### Field Descriptions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | enum | `"auto"` | BLAS backend: `"oktoblas"` for OktoBLAS, `"cublas"` for NVIDIA cuBLAS, `"auto"` for automatic selection |
| `precision` | enum | `"auto"` | Computation precision for matrix operations |
| `kernel` | enum | `"auto"` | Specific OktoBLAS kernel to use |
| `autoselect` | boolean | `true` | Automatically select best kernel for matrix size |
| `benchmark_mode` | boolean | `false` | Run benchmark to select optimal kernel |
| `streams` | number | `1` | Number of CUDA streams for parallel operations |
| `memory_pool` | enum | `"auto"` | Enable memory pooling for faster allocations |

---

## 🆕 ACCELERATE Block (Proposed for v1.3)

The `ACCELERATE` block provides high-level acceleration hints to the engine.

```ebnf
<accelerate_block> ::=
  "ACCELERATE" "{"
      [<accelerate_gemm>]
      [<accelerate_attention>]
      [<accelerate_softmax>]
      [<accelerate_layernorm>]
      [<accelerate_fused_ops>]
  "}"

<accelerate_gemm> ::=
  "gemm" ":" <acceleration_config>

<accelerate_attention> ::=
  "attention" ":" <acceleration_config>

<accelerate_softmax> ::=
  "softmax" ":" <acceleration_config>

<accelerate_layernorm> ::=
  "layernorm" ":" <acceleration_config>

<accelerate_fused_ops> ::=
  "fused_ops" ":" <boolean>

<acceleration_config> ::=
  ("oktoblas" | "native" | "auto" | "disabled")
```

### Example

```okt
ACCELERATE {
    gemm: "oktoblas"
    attention: "oktoblas"
    softmax: "oktoblas"
    layernorm: "native"
    fused_ops: true
}
```

### Use Case

This allows users to selectively enable OktoBLAS for specific operations:

```okt
# Use OktoBLAS only for GEMM (where it beats PyTorch)
ACCELERATE {
    gemm: "oktoblas"       # 125% PyTorch!
    attention: "native"     # Use native for now
    fused_ops: false
}
```

---

## 🆕 BENCHMARK Block (Proposed for v1.3)

The `BENCHMARK` block allows users to run performance benchmarks before training.

```ebnf
<benchmark_block> ::=
  "BENCHMARK" "{"
      [<benchmark_operations>]
      [<benchmark_sizes>]
      [<benchmark_iterations>]
      [<benchmark_warmup>]
      [<benchmark_compare>]
      [<benchmark_output>]
  "}"

<benchmark_operations> ::=
  "operations" ":" "[" <operation_list> "]"

<operation_list> ::=
  ("gemm_fp16" | "gemm_fp32" | "attention" | "softmax" | "layernorm")

<benchmark_sizes> ::=
  "matrix_sizes" ":" "[" <number_list> "]"

<benchmark_iterations> ::=
  "iterations" ":" <number>

<benchmark_warmup> ::=
  "warmup" ":" <number>

<benchmark_compare> ::=
  "compare_with" ":" "[" ("pytorch" | "cublas" | "cutlass") "]"

<benchmark_output> ::=
  "output" ":" <path>
```

### Example

```okt
BENCHMARK {
    operations: ["gemm_fp16", "gemm_fp32", "attention"]
    matrix_sizes: [1024, 2048, 4096]
    iterations: 100
    warmup: 10
    compare_with: ["pytorch"]
    output: "benchmark_results.json"
}
```

### Engine Behavior

When the engine encounters a `BENCHMARK` block:

1. Run benchmarks for each operation and size
2. Compare with specified frameworks
3. Save results to output file
4. **Auto-select optimal kernels** based on results
5. Log performance summary

---

## 🆕 TENSOR_CORES Block (Proposed for v1.3)

Control Tensor Core usage explicitly.

```ebnf
<tensor_cores_block> ::=
  "TENSOR_CORES" "{"
      [<tc_enabled>]
      [<tc_precision>]
      [<tc_min_size>]
  "}"

<tc_enabled> ::=
  "enabled" ":" <boolean>

<tc_precision> ::=
  "precision" ":" ("fp16" | "bf16" | "tf32")

<tc_min_size> ::=
  "min_matrix_size" ":" <number>
```

### Example

```okt
TENSOR_CORES {
    enabled: true
    precision: "fp16"
    min_matrix_size: 512  # Use Tensor Cores only for matrices >= 512
}
```

---

## 🆕 Enhanced ENV Block (v1.3)

Add OktoBLAS-specific fields to ENV:

```ebnf
<env_block_v13> ::=
  "ENV" "{"
      ... existing fields ...
      [<env_blas_backend>]
      [<env_tensor_cores>]
      [<env_cuda_streams>]
  "}"

<env_blas_backend> ::=
  "blas_backend" ":" ("oktoblas" | "cublas" | "auto")

<env_tensor_cores> ::=
  "tensor_cores" ":" ("enabled" | "disabled" | "auto")

<env_cuda_streams> ::=
  "cuda_streams" ":" <number>
```

### Example

```okt
ENV {
    accelerator: "gpu"
    min_memory: "8GB"
    precision: "fp16"
    blas_backend: "oktoblas"    # NEW: Use OktoBLAS
    tensor_cores: "enabled"      # NEW: Enable Tensor Cores
    cuda_streams: 4              # NEW: 4 parallel streams
}
```

---

## 🆕 Enhanced TRAIN Block (v1.3)

Add OktoBLAS control to TRAIN:

```ebnf
<train_block_v13> ::=
  "TRAIN" "{"
      ... existing fields ...
      [<train_gemm_kernel>]
      [<train_attention_kernel>]
  "}"

<train_gemm_kernel> ::=
  "gemm_kernel" ":" ("final_v1" | "best_v3" | "pure" | "auto")

<train_attention_kernel> ::=
  "attention_kernel" ":" ("flash_v2" | "standard" | "auto")
```

### Example

```okt
TRAIN {
    epochs: 10
    batch_size: 16
    learning_rate: 0.0001
    device: "cuda"
    
    # OktoBLAS-specific
    gemm_kernel: "final_v1"       # Use fastest kernel
    attention_kernel: "flash_v2"  # Use FlashAttention
}
```

---

## 🆕 MONITOR Metrics for OktoBLAS (v1.3)

New metrics available in MONITOR block:

```okt
MONITOR {
    metrics: [
        "loss",
        "accuracy",
        # New OktoBLAS metrics
        "gemm_tflops",           # GEMM TFLOPS achieved
        "attention_tflops",       # Attention TFLOPS
        "tensor_core_utilization", # % Tensor Core usage
        "blas_kernel_used",       # Which kernel is active
        "memory_bandwidth"        # GB/s bandwidth
    ]
    
    notify_if {
        gemm_tflops < 30
        tensor_core_utilization < 80%
    }
}
```

---

## Complete Example Script

```okt
# okto_version: "1.3"
PROJECT "OktoBLAS-Accelerated-Training"
DESCRIPTION "Training with OktoBLAS acceleration for maximum TFLOPS"

ENV {
    accelerator: "gpu"
    min_memory: "8GB"
    precision: "fp16"
    blas_backend: "oktoblas"
    tensor_cores: "enabled"
    cuda_streams: 4
}

# Run initial benchmark
BENCHMARK {
    operations: ["gemm_fp16", "attention"]
    matrix_sizes: [1024, 2048, 4096]
    iterations: 50
    compare_with: ["pytorch"]
    output: "benchmark_results.json"
}

# Configure BLAS
BLAS {
    backend: "oktoblas"
    precision: "fp16"
    kernel: "auto"        # Let engine pick best
    autoselect: true
    streams: 4
    memory_pool: "enabled"
}

# Accelerate specific operations
ACCELERATE {
    gemm: "oktoblas"      # 125% PyTorch for 1024-2048
    attention: "oktoblas" # 346% PyTorch for small seq
    softmax: "native"
    fused_ops: true
}

TENSOR_CORES {
    enabled: true
    precision: "fp16"
    min_matrix_size: 512
}

DATASET {
    train: "dataset/openorca.jsonl"
    format: "jsonl"
    type: "chat"
}

MODEL {
    base: "oktoseek/mini-gpt"
    architecture: "transformer"
    parameters: 64M
    precision: "fp16"
}

TRAIN {
    epochs: 3
    batch_size: 16
    learning_rate: 0.0001
    device: "cuda"
    gemm_kernel: "final_v1"
    attention_kernel: "flash_v2"
}

MONITOR {
    metrics: [
        "loss",
        "gemm_tflops",
        "attention_tflops",
        "tensor_core_utilization"
    ]
    
    notify_if {
        gemm_tflops < 30
        loss > 2.0
    }
    
    log_to: "logs/oktoblas_training.log"
}

EXPORT {
    format: ["safetensors", "gguf"]
    path: "output/"
    quantization: "fp16"
}
```

---

## Benefits of OktoBLAS Integration

| Feature | Benefit |
|---------|---------|
| **BLAS Backend Selection** | Use OktoBLAS where it beats PyTorch |
| **Kernel Selection** | Fine-tune performance per operation |
| **Benchmark Integration** | Auto-select optimal kernels |
| **Tensor Core Control** | Maximum FP16 performance |
| **Multi-Stream** | Parallel operations for higher throughput |
| **Performance Metrics** | Real-time TFLOPS monitoring |

---

## Implementation Notes

For OktoEngine implementation:

1. **Parse new blocks** in grammar
2. **Map to OktoBLAS API** in Rust backend
3. **Kernel dispatch** based on configuration
4. **Fallback to PyTorch/cuBLAS** if OktoBLAS unavailable
5. **Benchmark integration** for auto-tuning

---

**Status:** Implemented in v1.3  
**Release:** December 2025  
**Maintainer:** OktoSeek AI

---

© 2025 OktoSeek AI. All Rights Reserved.

