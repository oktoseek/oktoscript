# OktoScript – Frequently Asked Questions (FAQ)

Common questions and answers about OktoScript, a domain-specific language for AI training, evaluation, and deployment.

---

## 1. Even if FT_LORA already points to a base model and dataset, why must I still declare the MODEL and DATASET blocks?

**Answer:**

In OktoScript, `MODEL` and `DATASET` blocks define the **global context** of your project. They represent the default base configuration for the entire pipeline.

The `FT_LORA` block does not replace them—it only defines **how** the fine-tuning is performed. This explicit separation makes scripts clearer, more organized, and avoids hidden assumptions.

**Benefits of explicit declaration:**
- ✅ **Readability** - Anyone can understand the project structure at a glance
- ✅ **Debugging** - Clear separation of concerns makes troubleshooting easier
- ✅ **Reproducibility** - All configuration is visible and version-controlled
- ✅ **Documentation** - The script serves as self-documenting code

**Example:**
```okt
MODEL {
    base: "oktoseek/base-llm-7b"  # Global model context
}

DATASET {
    train: "dataset/main.jsonl"   # Global dataset context
}

FT_LORA {
    base_model: "oktoseek/base-llm-7b"  # Explicit for LoRA
    train_dataset: "dataset/main.jsonl"  # Explicit for LoRA
    lora_rank: 8
}
```

This design follows the principle: **explicit is better than implicit**, especially in AI pipelines where assumptions can lead to costly mistakes.

---

## 2. If I already use FT_LORA, why is the TRAIN block still mandatory?

**Answer:**

`FT_LORA` defines **what kind of training** happens (LoRA adapters), but `TRAIN` defines **how the training loop is executed** (optimizer, batch size, device, etc.).

**Think of it this way:**
- `TRAIN` = The engine (how training runs)
- `FT_LORA` = The driving mode (what gets trained)

**The TRAIN block controls:**
- Optimizer (adam, adamw, sgd, etc.)
- Batch size and gradient accumulation
- Device selection (cpu, cuda, mps)
- Learning rate and scheduler
- Training strategy (early stopping, checkpoints)

**Example:**
```okt
TRAIN {
    epochs: 5
    batch_size: 4
    optimizer: "adamw"
    learning_rate: 0.00003
    device: "cuda"
}

FT_LORA {
    lora_rank: 8
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj"]
}
```

Both blocks are required because they serve different purposes in the declarative DSL structure.

---

## 3. How do I define the final output of my model in OktoScript?

**Answer:**

The final output is always defined in the `EXPORT` block, regardless of whether you use `TRAIN` or `FT_LORA`.

**For standard training:**
```okt
EXPORT {
    format: ["gguf", "onnx", "okm"]
    path: "./export/"
}
```

**For LoRA fine-tuning:**
```okt
EXPORT {
    format: ["safetensors", "okm"]
    path: "./export/lora_patch/"
}
```

**What gets exported:**
- With `TRAIN`: Full model weights in specified formats
- With `FT_LORA`: LoRA adapter weights (safetensors) + optional merged model (okm)

The `EXPORT` block controls:
- ✅ Adapter generation (LoRA patches via safetensors)
- ✅ OktoSeek package generation (okm format)
- ✅ Cross-platform formats (onnx, gguf)
- ✅ Quantization settings

**Key point:** Export responsibility is clearly separated from training logic, keeping the DSL clean and modular.

---

## 4. What is the difference between FT_LORA and TRAIN blocks?

**Answer:**

| Block | Role | Purpose |
|-------|------|---------|
| `TRAIN` | Training loop configuration | Defines **how** training runs (optimizer, batch size, device) |
| `FT_LORA` | LoRA adapter configuration | Defines **what** gets trained (LoRA rank, alpha, target modules) |

**Important:** `FT_LORA` is **not** a replacement for `TRAIN`—it's an **extension** that modifies how training is applied to the model.

**When to use each:**
- **Use `TRAIN` alone:** Full fine-tuning of all model parameters
- **Use `TRAIN` + `FT_LORA`:** Efficient fine-tuning with LoRA adapters (recommended for large models)

**Example:**
```okt
# Full fine-tuning
TRAIN {
    epochs: 10
    batch_size: 32
    device: "cuda"
}

# LoRA fine-tuning (more efficient)
TRAIN {
    epochs: 5
    batch_size: 4
    device: "cuda"
}

FT_LORA {
    lora_rank: 8
    lora_alpha: 32
}
```

This separation keeps the language modular and scalable.

---

## 5. Do I need to repeat the base model inside FT_LORA if it is already declared in MODEL?

**Answer:**

**Yes, by design.** OktoScript prefers explicit declarations over implicit inference.

Even though the engine could technically infer the model from `MODEL`, keeping `base_model` inside `FT_LORA`:

- ✅ **Avoids ambiguity** - No guessing which model is used
- ✅ **Makes scripts self-contained** - Each block is independent
- ✅ **Improves readability** - Clear at a glance what's happening
- ✅ **Helps during audits** - Easier to review and validate

**Example:**
```okt
MODEL {
    base: "oktoseek/base-llm-7b"  # Global context
}

FT_LORA {
    base_model: "oktoseek/base-llm-7b"  # Explicit for LoRA
    lora_rank: 8
}
```

**This is an intentional design decision** to favor clarity and safety over convenience. In AI pipelines, explicit is safer than implicit.

---

## 6. What happens if I use both DATASET.train and mix_datasets at the same time?

**Answer:**

**Simple rule:** `mix_datasets` **overrides** `DATASET.train` when present.

**Priority order:**
1. `mix_datasets` in `FT_LORA` (highest priority)
2. `mix_datasets` in `DATASET` block
3. `DATASET.train` (default, lowest priority)

**Example:**
```okt
DATASET {
    train: "dataset/main.jsonl"  # Default dataset
}

FT_LORA {
    mix_datasets: [
        { path: "dataset/a.jsonl", weight: 70 },
        { path: "dataset/b.jsonl", weight: 30 }
    ]
    # This mix_datasets overrides DATASET.train
}
```

**Why this design?**
- Allows flexibility without breaking the main structure
- Enables dataset-specific configurations per training method
- Maintains backward compatibility with v1.0

**Best practice:** Use `DATASET.train` for the default, and `mix_datasets` when you need weighted mixing.

---

## 7. Does OktoScript replace Python?

**Answer:**

**No.** OktoScript does **not** replace Python. Instead, it replaces the **complex configuration boilerplate** typically written in Python.

**The relationship:**
- **Python** = Coding and programming (general-purpose language)
- **OktoScript** = Configuration of AI pipelines (domain-specific language)

**Think of it this way:**
```
Python (Engine) ← OktoScript (Configuration Layer) ← User
```

OktoScript sits **above** Python as a declarative layer, while Python powers the OktoEngine underneath.

**What OktoScript replaces:**
- ❌ Hundreds of lines of Python configuration code
- ❌ Complex YAML files with unclear structure
- ❌ Repetitive training scripts

**What Python still does:**
- ✅ Powers the OktoEngine
- ✅ Executes the training loop
- ✅ Handles low-level operations
- ✅ Provides hooks for custom logic

**Analogy:** OktoScript is to Python what Docker Compose is to Docker—a declarative configuration layer that simplifies complex operations.

---

## 8. Can I use multiple datasets with different weights?

**Answer:**

**Yes!** This is one of the key features of OktoScript v1.1.

**Syntax:**
```okt
DATASET {
    mix_datasets: [
        { path: "dataset/general.jsonl", weight: 60 },
        { path: "dataset/technical.jsonl", weight: 30 },
        { path: "dataset/creative.jsonl", weight: 10 }
    ]
    sampling: "weighted"
    shuffle: true
}
```

**Benefits:**
- ✅ **Balanced training** - Control dataset proportions
- ✅ **Domain blending** - Combine different data sources
- ✅ **Bias reduction** - Weight underrepresented data
- ✅ **Dataset prioritization** - Emphasize important data

**Rules:**
- Total weights must equal **exactly 100**
- `sampling: "weighted"` uses weights for sampling
- `sampling: "random"` ignores weights (uniform sampling)
- `shuffle: true` shuffles datasets before mixing

**Use case example:**
```okt
# Mix general conversations (60%) with technical Q&A (30%) and creative writing (10%)
mix_datasets: [
    { path: "dataset/conversations.jsonl", weight: 60 },
    { path: "dataset/technical_qa.jsonl", weight: 30 },
    { path: "dataset/creative.jsonl", weight: 10 }
]
```

---

## 9. What is the difference between EXPORT: safetensors and EXPORT: okm?

**Answer:**

| Format | Purpose | Use Case |
|--------|---------|----------|
| `safetensors` | Standard PyTorch weights format | LoRA adapters, model weights, HuggingFace compatibility |
| `okm` | OktoSeek optimized package | OktoSeek IDE, Flutter SDK, mobile apps, exclusive tools |
| `onnx` | Universal inference format | Production deployment, cross-platform compatibility |
| `gguf` | Local inference format | Ollama, Llama.cpp, local deployment |

**For LoRA fine-tuning:**
- `safetensors` → Saves only the LoRA adapter patch (small file, ~10-100MB)
- `okm` → Saves a full OktoSeek model package (includes adapter + metadata)

**Example:**
```okt
FT_LORA {
    lora_rank: 8
}

EXPORT {
    format: ["safetensors", "okm"]
    path: "./export/"
}
```

**Output:**
- `./export/adapter.safetensors` - LoRA adapter (for HuggingFace/PyTorch)
- `./export/model.okm` - OktoSeek package (for OktoSeek ecosystem)

**Why both?**
- `safetensors` for compatibility with standard ML tools
- `okm` for optimized OktoSeek ecosystem integration

---

## 10. Is OktoScript a programming language or a DSL?

**Answer:**

**OktoScript is a Domain-Specific Language (DSL).**

**What it is NOT:**
- ❌ A general-purpose programming language
- ❌ A scripting language with loops and variables
- ❌ A replacement for Python or JavaScript

**What it IS:**
- ✅ A declarative configuration language
- ✅ Purpose-built for AI pipelines
- ✅ Domain-specific (focused on AI training/deployment)

**Key characteristics:**
- **Declarative** - You describe **what** you want, not **how** to do it
- **No control flow** - No loops, conditionals, or functions
- **Block-based** - Configuration organized in semantic blocks
- **Type-safe** - Validated against grammar specification

**Why call it a DSL?**
- ✅ Technically accurate
- ✅ Increases professional credibility
- ✅ Sets correct expectations
- ✅ Distinguishes from general-purpose languages

**Analogy:** OktoScript is to AI pipelines what SQL is to databases—a specialized language for a specific domain.

---

## 11. What happens internally when I write FT_LORA?

**Answer:**

When you use `FT_LORA`, the OktoEngine performs these steps:

**1. Model Loading:**
- Loads the base model specified in `base_model`
- Initializes model architecture

**2. LoRA Adapter Injection:**
- Freezes the main model layers
- Adds LoRA adapters to selected modules (e.g., `q_proj`, `v_proj`)
- Adapters are low-rank matrices (rank × alpha)

**3. Training:**
- Trains **only** the LoRA adapter weights
- Main model weights remain frozen
- Uses optimizer and settings from `TRAIN` block

**4. Export:**
- Saves adapter weights via `EXPORT` block
- Optionally merges adapter into base model (if specified)

**Benefits:**
- ✅ **Reduced GPU usage** - Up to 90% less VRAM
- ✅ **Faster training** - Only small adapters are updated
- ✅ **Smaller files** - Adapter weights are tiny (~10-100MB)
- ✅ **Specialization** - Multiple adapters for different tasks
- ✅ **Flexibility** - Combine adapters at inference time

**Example flow:**
```
Base Model (7B params, frozen)
    ↓
+ LoRA Adapters (8 rank × 32 alpha = ~256 params per module)
    ↓
Training (only adapters updated)
    ↓
Export adapter.safetensors (~50MB)
```

---

## 12. Why is explicit declaration required instead of auto-inference?

**Answer:**

**Because transparency is better than hidden assumptions**, especially in AI pipelines.

**Problems with auto-inference:**
- ❌ Hidden assumptions can lead to silent mistakes
- ❌ Difficult to debug when things go wrong
- ❌ Unclear what the system is actually doing
- ❌ Harder to audit and review

**Benefits of explicit declaration:**
- ✅ **Self-documenting** - Scripts explain themselves
- ✅ **Auditable** - Easy to review and validate
- ✅ **Beginner-friendly** - Clear what's happening
- ✅ **Safe** - No hidden behavior or assumptions

**Example of explicit vs implicit:**
```okt
# Explicit (OktoScript style)
MODEL {
    base: "oktoseek/base-llm-7b"
}

FT_LORA {
    base_model: "oktoseek/base-llm-7b"  # Explicit, even if redundant
}

# Implicit (what we avoid)
FT_LORA {
    # base_model inferred from MODEL block - NOT in OktoScript
}
```

**Philosophy:** In AI, explicit is safer than implicit. A few extra lines of configuration prevent costly mistakes.

---

## 13. Can I run LoRA without EXPORT?

**Answer:**

**Technically yes, but it's not recommended.**

**What happens without EXPORT:**
- ✅ Training completes successfully
- ✅ Adapter weights are trained
- ❌ Adapter weights are **not saved**
- ❌ Training becomes useless after process ends

**Best practice:**
```okt
FT_LORA {
    lora_rank: 8
    lora_alpha: 32
}

EXPORT {
    format: ["safetensors", "okm"]
    path: "./export/"
}
```

**Why always include EXPORT:**
- ✅ Preserves your work
- ✅ Enables model reuse
- ✅ Allows deployment
- ✅ Supports version control

**Exception:** If you're only testing or debugging, you might skip EXPORT temporarily, but always add it before production training.

---

## 14. What if I want to merge a LoRA adapter into the final model later?

**Answer:**

**Current support (v1.1):**

You can merge LoRA adapters using OktoEngine's internal tools or Python hooks:

**Option 1: Using Hooks (Current)**
```okt
HOOKS {
    after_train: "scripts/merge_lora.py"
}
```

**Option 2: Manual merge with OktoEngine CLI**
```bash
okto_merge --adapter ./export/adapter.safetensors \
           --base ./models/base-model \
           --output ./export/merged-model
```

**Future support (v2.0+):**

A dedicated `MERGE` block is planned:

```okt
MERGE {
    source: "export/adapter.safetensors"
    target: "models/base-model"
    output: "export/merged-model"
    format: ["okm", "onnx"]
}
```

**Why merge?**
- ✅ Single model file (no separate adapter needed)
- ✅ Faster inference (no adapter loading)
- ✅ Easier deployment (one file instead of two)
- ✅ Better compatibility (works with standard tools)

**When to merge:**
- After training is complete
- Before deployment
- When you want a standalone model

---

## 15. Why choose OktoScript over YAML or Python scripts?

**Answer:**

**OktoScript is purpose-built for AI pipelines**, while YAML and Python are generic tools.

**Comparison:**

| Feature | OktoScript | YAML | Python |
|---------|------------|------|--------|
| **Purpose** | AI pipelines | Generic config | General programming |
| **Readability** | ✅ Block-based, semantic | ⚠️ Flat, no structure | ❌ Code complexity |
| **Validation** | ✅ Grammar-enforced | ⚠️ Manual validation | ❌ Runtime errors |
| **Type Safety** | ✅ Built-in | ❌ No types | ⚠️ Runtime checking |
| **AI-Specific** | ✅ LoRA, RAG, monitoring | ❌ Generic | ⚠️ Requires libraries |
| **Learning Curve** | ✅ Simple blocks | ⚠️ Syntax learning | ❌ Programming required |
| **IDE Support** | ✅ OktoSeek IDE | ⚠️ Generic editors | ✅ IDEs available |

**Key advantages of OktoScript:**

1. **Purpose-built for AI**
   - Native support for LoRA, RAG, monitoring
   - AI-specific blocks and concepts
   - Optimized for ML workflows

2. **Human-oriented**
   - Readable by non-programmers
   - Self-documenting structure
   - Clear semantic blocks

3. **Less error-prone**
   - Grammar validation
   - Type checking
   - Constraint enforcement

4. **Integrated ecosystem**
   - OktoSeek IDE support
   - OktoEngine integration
   - Flutter SDK compatibility

5. **Single config file**
   - Everything in one `.okt` file
   - No scattered configuration
   - Version control friendly

**Example comparison:**

**YAML (generic):**
```yaml
model:
  base: "oktoseek/base"
train:
  epochs: 5
  batch_size: 32
# No validation, no structure, unclear relationships
```

**Python (complex):**
```python
from transformers import Trainer, TrainingArguments
# 100+ lines of code
# Complex error handling
# Hard to read and maintain
```

**OktoScript (focused):**
```okt
MODEL {
    base: "oktoseek/base"
}

TRAIN {
    epochs: 5
    batch_size: 32
}
# Clear, validated, self-documenting
```

**Bottom line:** OktoScript is to AI pipelines what Docker Compose is to containers—a declarative DSL that simplifies complex operations.

---

## 16. How does OktoScript handle model versioning and checkpoints?

**Answer:**

OktoScript uses the `runs/` directory structure for automatic versioning and checkpoint management.

**Structure:**
```
runs/
  └── my-model/
      ├── checkpoint-100/
      │   └── model.safetensors
      ├── checkpoint-200/
      │   └── model.safetensors
      ├── tokenizer.json
      ├── training_logs.json
      └── metrics.json
```

**Checkpoint configuration:**
```okt
TRAIN {
    epochs: 10
    checkpoint_steps: 100  # Save every 100 steps
    checkpoint_path: "./checkpoints"
}
```

**Resume from checkpoint:**
```okt
TRAIN {
    resume_from_checkpoint: "./checkpoints/checkpoint-500"
    epochs: 10
}
```

**Benefits:**
- ✅ Automatic versioning by run name
- ✅ Step-based checkpointing
- ✅ Easy resume from any checkpoint
- ✅ Training logs and metrics per run

**Best practice:** Use descriptive project names in `PROJECT` block to organize runs.

---

## 17. Can I use custom Python code with OktoScript?

**Answer:**

**Yes!** OktoScript supports custom Python code through the `HOOKS` block.

**Available hooks:**
```okt
HOOKS {
    before_train: "scripts/preprocess.py"
    after_train: "scripts/postprocess.py"
    before_epoch: "scripts/custom_early_stop.py"
    after_epoch: "scripts/log_custom_metrics.py"
    on_checkpoint: "scripts/backup_checkpoint.sh"
    custom_metric: "scripts/toxicity_calculator.py"
}
```

**Hook script interface:**
```python
# scripts/preprocess.py
def before_train(config, dataset, model):
    # Custom preprocessing
    # Modify config if needed
    return config

# scripts/after_epoch.py
def after_epoch(epoch, metrics, model_state):
    # Custom logging, early stopping logic
    # Return True to stop training
    return False
```

**Use cases:**
- Custom data preprocessing
- Custom metrics calculation
- Custom early stopping logic
- External API integration
- Custom logging

**Key point:** OktoScript handles the configuration, Python handles the custom logic. Best of both worlds.

---

## 18. What happens if I specify conflicting configurations?

**Answer:**

OktoScript has **clear priority rules** to handle conflicts:

**Priority order (highest to lowest):**
1. Block-specific overrides (e.g., `mix_datasets` in `FT_LORA`)
2. Block-level settings (e.g., `FT_LORA` over `TRAIN` for LoRA)
3. Global settings (e.g., `DATASET.train`)

**Example conflicts and resolution:**

**Conflict 1: Dataset specification**
```okt
DATASET {
    train: "dataset/a.jsonl"  # Lower priority
}

FT_LORA {
    mix_datasets: [...]  # Higher priority - overrides DATASET.train
}
```
**Resolution:** `mix_datasets` is used, `DATASET.train` is ignored.

**Conflict 2: TRAIN vs FT_LORA**
```okt
TRAIN {
    epochs: 10
}

FT_LORA {
    epochs: 5  # This is used for LoRA training
}
```
**Resolution:** `FT_LORA.epochs` is used, but `TRAIN` optimizer/device settings still apply.

**Validation:**
- OktoEngine validates configurations before training
- Conflicts are reported with clear error messages
- Use `okto validate` to check before training

---

## 19. How do I debug an OktoScript file?

**Answer:**

**Step 1: Validate syntax**
```bash
okto validate train.okt
```

**Step 2: Check logs**
```okt
LOGGING {
    save_logs: true
    log_level: "debug"  # Enable debug logging
    log_every: 1
}
```

**Step 3: Use MONITOR for system diagnostics**
```okt
MONITOR {
    level: "full"
    log_system: ["gpu_memory_used", "cpu_usage", "temperature"]
    dashboard: true  # Real-time visualization
}
```

**Step 4: Check validation errors**
Common errors and solutions:
- `Dataset file not found` → Check file paths
- `Invalid optimizer` → Use allowed values (adam, adamw, sgd, etc.)
- `Model base not found` → Verify model path or HuggingFace name
- `Dataset mixing weights invalid` → Total must equal 100

**Step 5: Use system diagnostics**
```bash
okto_doctor  # Shows GPU, CUDA, RAM, drivers
```

**Best practices:**
- Always validate before training
- Start with `log_level: "debug"`
- Use `MONITOR` dashboard for real-time insights
- Check `runs/*/training_logs.json` for detailed logs

---

## 20. Is OktoScript production-ready?

**Answer:**

**Yes, OktoScript v1.1 is production-ready** for AI training and deployment pipelines.

**Production features:**
- ✅ **Stable grammar** - Well-defined and validated
- ✅ **Error handling** - Comprehensive validation
- ✅ **Monitoring** - System and training telemetry
- ✅ **Export formats** - Production-ready formats (ONNX, GGUF, OKM)
- ✅ **Deployment** - API, mobile, edge targets
- ✅ **Security** - Model encryption and watermarking
- ✅ **Logging** - Comprehensive logging and metrics

**Production checklist:**
```okt
PROJECT "ProductionModel"
VERSION "1.0"

# ... configuration ...

SECURITY {
    encrypt_model: true
    watermark: true
}

MONITOR {
    level: "full"
    dashboard: true
}

EXPORT {
    format: ["onnx", "okm"]  # Production formats
    optimize_for: "speed"
}

DEPLOY {
    target: "api"
    requires_auth: true
    max_concurrent_requests: 100
}
```

**Used by:**
- OktoSeek IDE (production)
- Research institutions
- AI development teams
- Educational platforms

**Version stability:**
- v1.0: Stable, production-ready
- v1.1: Backward compatible, adds LoRA and monitoring

---

## Need More Help?

- 📖 [Complete Grammar Specification](./grammar.md)
- 🚀 [Getting Started Guide](./GETTING_STARTED.md)
- ✅ [Validation Rules](../VALIDATION_RULES.md)
- 💡 [Examples](../examples/)
- 🐛 [Troubleshooting](./grammar.md#troubleshooting)

**Still have questions?** Open an issue on [GitHub](https://github.com/oktoseek/oktoscript/issues) or contact **service@oktoseek.com**.

---

**OktoScript** is developed and maintained by **OktoSeek AI**.



