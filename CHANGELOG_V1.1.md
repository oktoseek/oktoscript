# OktoScript v1.1 Changelog

**Release Date:** November 2025  
**Status:** 100% Backward Compatible with v1.0

---

## 🎉 New Features

### 1. LoRA Fine-Tuning Support

Added `FT_LORA` block for efficient fine-tuning using Low-Rank Adaptation adapters.

**Benefits:**
- ✅ Reduced memory footprint (up to 90% less VRAM)
- ✅ Faster training times
- ✅ Smaller model files (only adapter weights)
- ✅ Easy to combine multiple LoRA adapters

**Example:**
```okt
# okto_version: "1.1"
FT_LORA {
    base_model: "oktoseek/base-llm-7b"
    lora_rank: 8
    lora_alpha: 32
    epochs: 5
    batch_size: 4
    learning_rate: 0.00003
    device: "cuda"
    target_modules: ["q_proj", "v_proj"]
}
```

**See:** [`examples/lora-finetuning.okt`](./examples/lora-finetuning.okt)

---

### 2. Dataset Mixing and Sampling

Enhanced `DATASET` block with support for mixing multiple datasets with weighted sampling.

**New Fields:**
- `mix_datasets`: Array of `{path, weight}` objects
- `dataset_percent`: Limit dataset usage (1-100)
- `sampling`: `"weighted"` or `"random"`
- `shuffle`: Shuffle datasets before mixing

**Example:**
```okt
DATASET {
    mix_datasets: [
        { path: "dataset/base.jsonl", weight: 70 },
        { path: "dataset/extra.jsonl", weight: 30 }
    ]
    dataset_percent: 80
    sampling: "weighted"
    shuffle: true
}
```

**Benefits:**
- ✅ Combine multiple datasets intelligently
- ✅ Control dataset proportions
- ✅ Limit dataset size for faster iteration
- ✅ Weighted or random sampling strategies

**See:** [`examples/dataset-mixing.okt`](./examples/dataset-mixing.okt)

---

### 3. Advanced System Monitoring

Added `MONITOR` block for comprehensive system and training telemetry.

**Features:**
- System metrics (GPU, CPU, RAM, temperature)
- Training speed metrics (tokens/s, samples/s)
- Real-time dashboard (optional)
- Configurable refresh intervals
- Export to JSON

**Example:**
```okt
MONITOR {
    level: "full"
    log_metrics: ["loss", "accuracy", "perplexity"]
    log_system: ["gpu_memory_used", "cpu_usage", "temperature"]
    log_speed: ["tokens_per_second", "samples_per_second"]
    refresh_interval: 2s
    export_to: "runs/logs/system.json"
    dashboard: true
}
```

**Benefits:**
- ✅ Monitor system resources during training
- ✅ Detect bottlenecks and optimize
- ✅ Track training speed
- ✅ Real-time visualization

---

### 4. Version Declaration

Added optional version declaration at the top of `.okt` files.

**Syntax:**
```okt
# okto_version: "1.1"
PROJECT "MyModel"
...
```

**Rules:**
- Optional (defaults to v1.0 if missing)
- Must be first line (comments allowed before)
- Format: `# okto_version: "1.1"` or `# okto_version: "1.0"`
- Enables v1.1 features when set to "1.1"

---

## 📁 New Optional Folders

v1.1 introduces optional folders for new features:

```
/runs/
  └── my-model/
      ├── logs/
      │   └── system.json      # MONITOR output
      └── lora/                 # LoRA adapters
          └── adapter.safetensors
```

**Note:** These folders are created automatically when using v1.1 features. Existing v1.0 structure remains unchanged.

---

## 🔄 Backward Compatibility

**100% Compatible with v1.0:**

- ✅ All v1.0 files work without modification
- ✅ v1.0 syntax remains valid
- ✅ No breaking changes
- ✅ Default version is v1.0 (if version not specified)

**Migration:**
- No migration required
- Simply add `# okto_version: "1.1"` to use new features
- Existing v1.0 files continue to work

---

## 📚 Documentation Updates

- ✅ [`docs/grammar.md`](./docs/grammar.md) - Updated with v1.1 grammar
- ✅ [`VALIDATION_RULES.md`](./VALIDATION_RULES.md) - Added v1.1 validation rules
- ✅ [`README.md`](./README.md) - Added v1.1 examples and features
- ✅ New examples in [`examples/`](./examples/)

---

## 🐛 Bug Fixes

None (this is a feature release)

---

## 🔮 Future Roadmap

Planned for future versions:
- Multi-GPU training support
- Distributed training
- Advanced quantization options
- More dataset formats
- Custom loss functions

---

**For questions or feedback:** [GitHub Issues](https://github.com/oktoseek/oktoscript/issues)



