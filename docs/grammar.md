# OktoScript Grammar Specification v1.2

Complete formal grammar for the OktoScript language, developed by **OktoSeek AI**.

> **Version Compatibility:** This specification covers OktoScript v1.2, which is 100% backward compatible with v1.0 and v1.1. Files without version declaration default to v1.0.

---

## Table of Contents

1. [Grammar Overview](#grammar-overview)
2. [Basic Metadata Blocks](#basic-metadata-blocks)
3. [ENV Block](#env-block)
4. [DATASET Block](#dataset-block)
5. [MODEL Block](#model-block)
6. [TRAIN Block](#train-block)
7. [METRICS Block](#metrics-block)
8. [VALIDATION Block](#validation-block)
9. [INFERENCE Block](#inference-block)
10. [CONTROL Block — Decision Engine](#control-block--decision-engine)
11. [MONITOR Block — Full Metrics Support](#monitor-block--full-metrics-support)
12. [GUARD Block — Safety / Ethics / Protection](#guard-block--safety--ethics--protection)
13. [BEHAVIOR Block — Model Personality](#behavior-block--model-personality)
14. [EXPLORER Block — Parameter Search](#explorer-block--parameter-search)
15. [STABILITY Block — Training Safety](#stability-block--training-safety)
16. [Boolean Support](#boolean-support)
17. [EXPORT Block](#export-block)
18. [DEPLOY Block](#deploy-block)
19. [SECURITY Block](#security-block)
20. [LOGGING Block](#logging-block)
21. [Model Inheritance](#model-inheritance)
22. [Extension Points & Hooks](#extension-points--hooks)
23. [Validation Rules](#validation-rules)
24. [Troubleshooting](#troubleshooting)
25. [Terminal / Basic Types](#terminal--basic-types)
26. [Full Script Example](#full-script-example)

---

## Grammar Overview

```ebnf
<oktoscript> ::=
  [<version_declaration>]
  <project_block>
  [<description_block>]
  [<version_block>]
  [<tags_block>]
  [<author_block>]
  [<env_block>]
  <dataset_block>
  <model_block>
  [<train_block> | <ft_lora_block>]
  [<metrics_block>]
  [<validation_block>]
  [<inference_block>]
  [<export_block>]
  [<deploy_block>]
  [<security_block>]
  [<logging_block>]
  [<monitor_block>]
  [<control_block>]
  [<guard_block>]
  [<behavior_block>]
  [<explorer_block>]
  [<stability_block>]
  [<hooks_block>]
```

**Note:** `TRAIN` and `FT_LORA` are mutually exclusive. Use `FT_LORA` for LoRA-based fine-tuning, or `TRAIN` for full fine-tuning.

**Required blocks:** PROJECT, DATASET, MODEL, TRAIN  
**Optional blocks:** ENV, DESCRIPTION, VERSION, TAGS, AUTHOR, and all others

---

## Version Declaration (v1.1+)

```ebnf
<version_declaration> ::=
  "# okto_version:" <string>
```

**Example:**
```okt
# okto_version: "1.2"
PROJECT "MyModel"
...
```

**Rules:**
- Optional at the top of the file
- If missing, defaults to v1.0
- Must be the first line (comments allowed before)
- Format: `# okto_version: "1.2"`, `# okto_version: "1.1"`, or `# okto_version: "1.0"`

---

## Basic Metadata Blocks

### PROJECT Block

```ebnf
<project_block> ::= 
  "PROJECT" <string>
```

**Constraints:**
- Project name must be a valid string (1-100 characters)
- Cannot contain special characters: `{`, `}`, `[`, `]`, `:`, `"`

**Example:**
```okt
PROJECT "PizzaBot"
```

### DESCRIPTION Block

```ebnf
<description_block> ::= 
  "DESCRIPTION" <string>
```

**Constraints:**
- Maximum 500 characters
- Can contain any UTF-8 characters

**Example:**
```okt
DESCRIPTION "AI specialized in pizza restaurant service"
```

### VERSION Block

```ebnf
<version_block> ::= 
  "VERSION" <string>
```

**Constraints:**
- Must follow semantic versioning (e.g., "1.0.0", "2.1.3")
- Format: `major.minor.patch` or `major.minor`

**Example:**
```okt
VERSION "1.0"
VERSION "2.1.3"
```

### TAGS Block

```ebnf
<tags_block> ::= 
  "TAGS" "[" <string_list> "]"
```

**Constraints:**
- Maximum 10 tags
- Each tag: 1-50 characters
- Tags are case-insensitive

**Example:**
```okt
TAGS ["food", "restaurant", "chatbot"]
```

### AUTHOR Block

```ebnf
<author_block> ::= 
  "AUTHOR" <string>
```

**Example:**
```okt
AUTHOR "OktoSeek"
```

---

## ENV Block

The `ENV` block defines environment requirements, hardware expectations, and execution preferences for OktoEngine. It is fully abstract and does not expose underlying implementation details (Python, PyTorch, TensorFlow, etc.). OktoEngine uses this block to configure the execution environment before running any training or inference operations.

**Purpose:**
- Define minimum environment requirements for a project
- Specify hardware preferences (CPU, GPU, TPU)
- Set memory and precision requirements
- Configure execution backend preferences
- Enable automatic dependency installation
- Specify platform and network requirements

**Note:** ENV is not a dependency list. It is a high-level execution requirement description that allows OktoEngine to decide how to configure the real execution environment.

### ENV Block Syntax

```ebnf
<env_block> ::=
  "ENV" "{"
      [<env_accelerator>]
      [<env_min_memory>]
      [<env_precision>]
      [<env_backend>]
      [<env_install_missing>]
      [<env_platform>]
      [<env_network>]
  "}"

<env_accelerator> ::=
  "accelerator" ":" ("auto" | "cpu" | "gpu" | "tpu")

<env_min_memory> ::=
  "min_memory" ":" <memory_string>

<memory_string> ::=
  "4GB" | "8GB" | "16GB" | "32GB" | "64GB"

<env_precision> ::=
  "precision" ":" ("auto" | "fp16" | "fp32" | "bf16")

<env_backend> ::=
  "backend" ":" ("auto" | "oktoseek")

<env_install_missing> ::=
  "install_missing" ":" ("true" | "false")

<env_platform> ::=
  "platform" ":" ("windows" | "linux" | "mac" | "any")

<env_network> ::=
  "network" ":" ("online" | "offline" | "required")
```

### ENV Block Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `accelerator` | enum | ❌ No | `"auto"` | Preferred compute unit: `"auto"`, `"cpu"`, `"gpu"`, `"tpu"` |
| `min_memory` | string | ❌ No | `"8GB"` | Required minimum RAM: `"4GB"`, `"8GB"`, `"16GB"`, `"32GB"`, `"64GB"` |
| `precision` | enum | ❌ No | `"auto"` | Numerical precision: `"auto"`, `"fp16"`, `"fp32"`, `"bf16"` |
| `backend` | enum | ❌ No | `"auto"` | Execution engine: `"auto"`, `"oktoseek"` |
| `install_missing` | boolean | ❌ No | `false` | If `true`, engine attempts automatic dependency installation |
| `platform` | enum | ❌ No | `"any"` | Target OS: `"windows"`, `"linux"`, `"mac"`, `"any"` |
| `network` | enum | ❌ No | `"online"` | Internet requirement: `"online"`, `"offline"`, `"required"` |

### ENV Block Examples

**Minimal ENV (uses defaults):**
```okt
ENV {
  accelerator: "gpu"
  min_memory: "8GB"
}
```

**Complete ENV configuration:**
```okt
ENV {
  accelerator: "gpu"
  min_memory: "16GB"
  precision: "fp16"
  backend: "oktoseek"
  install_missing: true
  platform: "any"
  network: "online"
}
```

**CPU-only training:**
```okt
ENV {
  accelerator: "cpu"
  min_memory: "8GB"
  precision: "fp32"
  install_missing: true
}
```

**Offline execution:**
```okt
ENV {
  accelerator: "gpu"
  min_memory: "16GB"
  network: "offline"
  install_missing: false
}
```

### ENV Block Constraints

1. **Memory format:** Must use `GB` suffix (e.g., `"8GB"`, not `"8"` or `"8 GB"`)
2. **Enum values:** Only predefined values are allowed
3. **Boolean values:** Must be `true` or `false` (lowercase)
4. **String values:** Must be quoted

### ENV Block Validation Rules

1. If `accelerator = "gpu"` and `min_memory < "8GB"` → **warning** (GPU training typically requires at least 8GB)
2. If `network = "offline"` → export formats like `onnx` or `gguf` are allowed (pre-downloaded models)
3. If `backend = "oktoseek"` → preferred default for OktoSeek ecosystem
4. If `install_missing = true` → engine must attempt auto-setup of missing dependencies
5. If no ENV block exists → defaults to:
   ```okt
   ENV {
     accelerator: "auto"
     min_memory: "8GB"
     backend: "auto"
   }
   ```

### Engine Behavior

When OktoEngine encounters an ENV block, it must:

1. **Read ENV block first:** Before any other stage (dataset loading, model initialization, etc.)
2. **Check system compatibility:** Verify RAM, GPU availability, platform, etc.
3. **Return detailed errors:** If system is incompatible, return specific error messages
4. **Auto-install dependencies:** If `install_missing: true`, attempt automatic setup
5. **Generate environment report:** Log analysis to `runs/{model}/env_report.json`

**Example env_report.json:**
```json
{
  "gpu_found": true,
  "gpu_name": "NVIDIA RTX 3090",
  "ram": "32GB",
  "ram_available": "28GB",
  "platform": "linux",
  "status": "compatible",
  "auto_install": true,
  "warnings": []
}
```

---

## DATASET Block

```ebnf
<dataset_block> ::= 
  "DATASET" "{" 
      [<dataset_train> | <mix_datasets>]
      [<dataset_validation>]
      [<dataset_test>]
      [<dataset_format>]
      [<dataset_type>]
      [<dataset_language>]
      [<dataset_augmentation>]
      [<dataset_percent>]
      [<dataset_sampling>]
      [<dataset_shuffle>]
      [<dataset_input_field>]
      [<dataset_output_field>]
  "}"

<dataset_train> ::=
  "train" ":" <path>

<dataset_validation> ::=
  "validation" ":" <path>

<dataset_test> ::=
  "test" ":" <path>

<dataset_format> ::= 
  "format" ":" ("jsonl" | "csv" | "txt" | "parquet" | "image+caption" | "qa" | "instruction" | "multimodal")

<dataset_type> ::=
  "type" ":" ("classification" | "generation" | "qa" | "chat" | "vision" | "regression")

<dataset_language> ::=
  "language" ":" ("en" | "pt" | "es" | "fr" | "multilingual")

<dataset_augmentation> ::=
  "augmentation" ":" "[" <string_list> "]"

<dataset_percent> ::=
  "dataset_percent" ":" <number>

<mix_datasets> ::=
  "mix_datasets" ":" "[" <mix_dataset_list> "]"

<mix_dataset_list> ::=
  <mix_dataset_item> { "," <mix_dataset_item> }

<mix_dataset_item> ::=
  "{" "path" ":" <path> "," "weight" ":" <number> "}"

<dataset_sampling> ::=
  "sampling" ":" ("weighted" | "random")

<dataset_shuffle> ::=
  "shuffle" ":" ("true" | "false")

<dataset_input_field> ::=
  "input_field" ":" <string>

<dataset_output_field> ::=
  ("output_field" | "target_field") ":" <string>

<dataset_context_fields> ::=
  "context_fields" ":" "[" <string_list> "]"
```

**Allowed augmentation values:**
- `"flip"` - Horizontal/vertical flip
- `"rotate"` - Random rotation
- `"brightness"` - Brightness adjustment
- `"contrast"` - Contrast adjustment
- `"noise"` - Add noise
- `"crop"` - Random cropping
- `"translate"` - Translation

**Validation Rules:**
- `train` path must exist and be readable
- File format must match declared `format`
- For `image+caption`, path must be a directory
- For JSONL/CSV, path must be a file

**Example (v1.0):**
```okt
DATASET {
    train: "dataset/train.jsonl"
    validation: "dataset/val.jsonl"
    test: "dataset/test.jsonl"
    format: "jsonl"
    type: "chat"
    language: "en"
    augmentation: ["flip", "rotate", "brightness"]
}
```

**Example (v1.1 - Dataset Mixing):**
```okt
DATASET {
    mix_datasets: [
        { path: "dataset/base.jsonl", weight: 70 },
        { path: "dataset/extra.jsonl", weight: 30 }
    ]
    dataset_percent: 50
    sampling: "weighted"
    shuffle: true
    format: "jsonl"
    type: "chat"
}
```

**Example (v1.2 - Custom Field Names):**
```okt
DATASET {
    train: "dataset/train.jsonl"
    validation: "dataset/val.jsonl"
    format: "jsonl"
    type: "chat"
    input_field: "input"
    output_field: "target"
}
```

**Example (v1.2 - With Context Fields):**
```okt
DATASET {
    train: "dataset/pizzaria.jsonl"
    validation: "dataset/val.jsonl"
    format: "jsonl"
    type: "chat"
    input_field: "input"
    output_field: "target"
    context_fields: ["menu", "drinks", "promotions"]
}
```

**Dataset JSONL with context:**
```jsonl
{"input": "What pizzas do you have?", "target": "We have Margherita, Pepperoni, and Four Cheese.", "menu": "Margherita: $34, Pepperoni: $39, Four Cheese: $45", "drinks": "Coke, Sprite, Water"}
{"input": "Do you have drinks?", "target": "Yes, we have Coke, Sprite, and Water.", "menu": "Margherita: $34, Pepperoni: $39", "drinks": "Coke, Sprite, Water"}
```

The context fields will be automatically included in the prompt:
- Input: `menu: Margherita: $34, Pepperoni: $39 | drinks: Coke, Sprite, Water | What pizzas do you have?`
- Target: `We have Margherita, Pepperoni, and Four Cheese.`

**Field Name Resolution (v1.2+):**
- If `input_field` and `output_field` are specified, use those exact field names
- If not specified, defaults are tried in order:
  1. `"input"` + `"output"` (standard format)
  2. `"input"` + `"target"` (common alternative)
  3. `"text"` (single field, used for both input and output)
  4. First string field in dataset (fallback)
- `context_fields` are optional and will be included in the prompt if present
- This ensures backward compatibility while allowing full customization

**Dataset Mixing Rules:**
- If `mix_datasets` is specified, it overrides `train`
- Total weights in `mix_datasets` must equal 100
- `dataset_percent` limits total dataset usage (1-100)
- `sampling: "weighted"` uses weights for sampling, `"random"` ignores weights
- `shuffle` controls whether datasets are shuffled before mixing

---

## MODEL Block

```ebnf
<model_block> ::= 
  "MODEL" "{"
      [<model_name>]
      <model_base>
      [<model_architecture>]
      [<model_parameters>]
      [<model_context_window>]
      [<model_precision>]
      [<model_inherit>]
      [<model_device>]
      [<adapter_block>]
  "}"

<model_name> ::=
  "name" ":" <string>

<model_base> ::=
  "base" ":" <string>

<model_architecture> ::=
  "architecture" ":" ("transformer" | "cnn" | "rnn" | "diffusion" | "vision-transformer" | "bert" | "gpt" | "t5")

<model_parameters> ::=
  "parameters" ":" <number> ("M" | "B" | "K")

<model_context_window> ::=
  "context_window" ":" <number>

<model_precision> ::=
  "precision" ":" ("fp32" | "fp16" | "int8" | "int4")

<model_inherit> ::=
  "inherit" ":" <string>

<model_device> ::=
  "device" ":" ("cuda" | "cpu" | "mps" | "auto")

<adapter_block> ::=
  "ADAPTER" "{"
      <adapter_type>
      <adapter_path>
      [<adapter_rank>]
      [<adapter_alpha>]
  "}"

<adapter_type> ::=
  "type" ":" ("lora" | "qlora" | "adapter" | "peft")

<adapter_path> ::=
  "path" ":" <path>

<adapter_rank> ::=
  "rank" ":" <number>

<adapter_alpha> ::=
  "alpha" ":" <number>
```

**Model Inheritance:**
- `inherit` allows reusing configuration from another model
- Inherited model must be defined in the same project or imported
- Child model can override any parent field
- Example: `inherit: "base-transformer"` loads base config, then applies current block

**Allowed base model formats:**
- HuggingFace format: `"username/model-name"`
- OktoSeek format: `"oktoseek/model-name"`
- Local path: `"./models/my-model"`
- URL: `"https://example.com/model"`

**Parameter constraints:**
- `parameters`: Must be positive number with suffix (K, M, B)
- `context_window`: Must be power of 2 (128, 256, 512, 1024, 2048, 4096, 8192)
- `precision`: Must match device capabilities

**Example:**
```okt
MODEL {
    name: "oktogpt"
    base: "oktoseek/pizza-small"
    architecture: "transformer"
    parameters: 120M
    context_window: 2048
    precision: "fp16"
    device: "cuda"
}
```

**Example with ADAPTER (LoRA/PEFT support):**
```okt
MODEL {
    name: "oktogpt"
    base: "google/flan-t5-base"
    device: "cuda"
    
    ADAPTER {
        type: "lora"
        path: "D:/model_trainee/phase1_sharegpt/ep2"
        rank: 16
        alpha: 32
    }
}
```

**Example with inheritance:**
```okt
# Base model definition
MODEL "base-transformer" {
    architecture: "transformer"
    context_window: 2048
    precision: "fp16"
}

# Child model inheriting from base
MODEL {
    inherit: "base-transformer"
    base: "oktoseek/custom-model"
    parameters: 250M
}
```

**ADAPTER Block:**
The `ADAPTER` sub-block enables parameter-efficient fine-tuning methods such as LoRA, QLoRA, PEFT, or other adapters. If an `ADAPTER` is defined, it is applied after the base model is loaded by the engine.

**Adapter constraints:**
- `type`: Must be one of `"lora"`, `"qlora"`, `"adapter"`, or `"peft"`
- `path`: Must point to a valid adapter directory or file
- `rank`: Optional, typically 4, 8, 16, 32, or 64 (for LoRA)
- `alpha`: Optional, typically 16, 32, or 64 (for LoRA scaling)

---

## FT_LORA Block (v1.1+)

Fine-tuning using LoRA (Low-Rank Adaptation) adapters. This block is an alternative to `TRAIN` for efficient fine-tuning.

```ebnf
<ft_lora_block> ::=
  "FT_LORA" "{"
      <ft_lora_base_model>
      <ft_lora_train_dataset>
      <ft_lora_rank>
      <ft_lora_alpha>
      [<ft_lora_dataset_percent>]
      [<ft_lora_mix_datasets>]
      [<ft_lora_epochs>]
      [<ft_lora_batch_size>]
      [<ft_lora_learning_rate>]
      [<ft_lora_device>]
      [<ft_lora_target_modules>]
  "}"

<ft_lora_base_model> ::=
  "base_model" ":" <string>

<ft_lora_train_dataset> ::=
  "train_dataset" ":" <path>

<ft_lora_rank> ::=
  "lora_rank" ":" <number>

<ft_lora_alpha> ::=
  "lora_alpha" ":" <number>

<ft_lora_dataset_percent> ::=
  "dataset_percent" ":" <number>

<ft_lora_mix_datasets> ::=
  "mix_datasets" ":" "[" <mix_dataset_list> "]"

<ft_lora_epochs> ::=
  "epochs" ":" <number>

<ft_lora_batch_size> ::=
  "batch_size" ":" <number>

<ft_lora_learning_rate> ::=
  "learning_rate" ":" <decimal>

<ft_lora_device> ::=
  "device" ":" ("cpu" | "cuda" | "mps" | "auto")

<ft_lora_target_modules> ::=
  "target_modules" ":" "[" <string_list> "]"
```

**Constraints:**
- `lora_rank`: Must be > 0 and <= 256 (typical: 4, 8, 16, 32)
- `lora_alpha`: Must be > 0 (typical: 16, 32, 64)
- `dataset_percent`: Must be 1-100
- If `mix_datasets` is specified, it overrides `train_dataset`
- Total weights in `mix_datasets` must equal 100

**Example:**
```okt
FT_LORA {
    base_model: "oktoseek/base-mini"
    train_dataset: "dataset/main.jsonl"
    lora_rank: 4
    lora_alpha: 16
    dataset_percent: 50
    mix_datasets: [
        { path: "dataset/base.jsonl", weight: 70 },
        { path: "dataset/extra.jsonl", weight: 30 }
    ]
    epochs: 3
    batch_size: 16
    learning_rate: 0.00003
    device: "cuda"
    target_modules: ["q_proj", "v_proj"]
}
```

**When to use FT_LORA vs TRAIN:**
- **FT_LORA**: Efficient fine-tuning, smaller memory footprint, faster training, good for domain adaptation
- **TRAIN**: Full fine-tuning, maximum flexibility, better for large architectural changes

---

## TRAIN Block

```ebnf
<train_block> ::=
  "TRAIN" "{"
      <train_epochs>
      <train_batch>
      [<train_lr>]
      [<train_optimizer>]
      [<train_scheduler>]
      <train_device>
      [<gradient_accumulation>]
      [<early_stopping>]
      [<checkpoint_steps>]
      [<checkpoint_path>]
      [<resume_from_checkpoint>]
      [<loss_function>]
      [<weight_decay>]
      [<gradient_clip>]
      [<warmup_steps>]
      [<save_strategy>]
      [<logging_steps>]
      [<save_steps>]
  "}"

<train_epochs> ::= 
  "epochs" ":" <number>

<train_batch> ::= 
  "batch_size" ":" <number>

<train_lr> ::= 
  "learning_rate" ":" <decimal>

<train_optimizer> ::= 
  "optimizer" ":" ( "adam" | "adamw" | "sgd" | "rmsprop" | "adafactor" | "lamb" )

<train_scheduler> ::= 
  "scheduler" ":" ("linear" | "cosine" | "cosine_with_restarts" | "polynomial" | "constant" | "constant_with_warmup" | "step")

<train_device> ::= 
  "device" ":" ("cpu" | "cuda" | "mps" | "auto")

<gradient_accumulation> ::= 
  "gradient_accumulation" ":" <number>

<early_stopping> ::= 
  "early_stopping" ":" ("true" | "false")

<checkpoint_steps> ::= 
  "checkpoint_steps" ":" <number>

<checkpoint_path> ::=
  "checkpoint_path" ":" <path>

<resume_from_checkpoint> ::=
  "resume_from_checkpoint" ":" <path>

<loss_function> ::=
  "loss" ":" ("cross_entropy" | "mse" | "mae" | "bce" | "focal" | "huber" | "kl_divergence")

<weight_decay> ::=
  "weight_decay" ":" <decimal>

<gradient_clip> ::=
  "gradient_clip" ":" <decimal>

<warmup_steps> ::=
  "warmup_steps" ":" <number>

<save_strategy> ::=
  "save_strategy" ":" ("steps" | "epoch" | "no")

<logging_steps> ::=
  "logging_steps" ":" <number>

<save_steps> ::=
  "save_steps" ":" <number>
```

**Allowed values and constraints:**

**Optimizers:**
- `adam` - Adam optimizer (default)
- `adamw` - Adam with weight decay
- `sgd` - Stochastic Gradient Descent
- `rmsprop` - RMSprop optimizer
- `adafactor` - Adafactor (memory efficient)
- `lamb` - LAMB optimizer (for large batches)

**Schedulers:**
- `linear` - Linear decay
- `cosine` - Cosine annealing
- `cosine_with_restarts` - Cosine with restarts
- `polynomial` - Polynomial decay
- `constant` - Constant learning rate
- `constant_with_warmup` - Constant with warmup
- `step` - Step decay

**Loss functions:**
- `cross_entropy` - Cross-entropy loss (classification)
- `mse` - Mean Squared Error (regression)
- `mae` - Mean Absolute Error (regression)
- `bce` - Binary Cross-Entropy
- `focal` - Focal loss (imbalanced data)
- `huber` - Huber loss (robust regression)
- `kl_divergence` - KL divergence

**Constraints:**
- `epochs`: Must be > 0 and <= 1000
- `batch_size`: Must be > 0 and <= 1024
- `learning_rate`: Must be > 0 and <= 1.0
- `gradient_accumulation`: Must be >= 1
- `checkpoint_steps`: Must be > 0
- `weight_decay`: Must be >= 0 and <= 1.0
- `gradient_clip`: Must be > 0

**Example:**
```okt
TRAIN {
    epochs: 10
    batch_size: 32
    learning_rate: 0.00025
    optimizer: "adamw"
    scheduler: "cosine"
    loss: "cross_entropy"
    device: "cuda"
    gradient_accumulation: 2
    early_stopping: true
    checkpoint_steps: 100
    checkpoint_path: "./checkpoints"
    weight_decay: 0.01
    gradient_clip: 1.0
    warmup_steps: 500
    save_strategy: "steps"
    logging_steps: 5    # Log every 5 steps (default: 10)
    save_steps: 500     # Save checkpoint every 500 steps (default: 500)
}
```

**Example with checkpoint resume:**
```okt
TRAIN {
    epochs: 20
    batch_size: 16
    learning_rate: 0.0001
    optimizer: "adamw"
    device: "cuda"
    resume_from_checkpoint: "./checkpoints/checkpoint-500"
    checkpoint_steps: 100
}
```

---

## METRICS Block

```ebnf
<metrics_block> ::= 
  "METRICS" "{"
      { <metric> | <custom_metric> }
  "}"

<metric> ::= 
  "accuracy" |
  "loss" |
  "perplexity" |
  "f1" |
  "f1_macro" |
  "f1_micro" |
  "f1_weighted" |
  "bleu" |
  "rouge" |
  "rouge_l" |
  "rouge_1" |
  "rouge_2" |
  "mae" |
  "mse" |
  "rmse" |
  "cosine_similarity" |
  "token_efficiency" |
  "response_coherence" |
  "hallucination_score" |
  "precision" |
  "recall" |
  "confusion_matrix"

<custom_metric> ::= 
  "custom" <string>
```

**Metric-specific constraints:**
- `accuracy`: Only for classification tasks
- `perplexity`: Only for language models
- `bleu`, `rouge`: Only for generation/translation tasks
- `mae`, `mse`, `rmse`: Only for regression tasks
- `confusion_matrix`: Only for classification, generates full matrix

**Example:**
```okt
METRICS {
    accuracy
    loss
    perplexity
    f1
    f1_macro
    rouge_l
    cosine_similarity
    custom "toxicity_score"
    custom "context_alignment"
}
```

---

## VALIDATION Block

```ebnf
<validation_block> ::=
  "VALIDATE" "{"
      [ "on_train" ":" ("true" | "false") ]
      [ "on_validation" ":" ("true" | "false") ]
      [ "frequency" ":" <number> ]
      [ "save_best_model" ":" ("true" | "false") ]
      [ "metric_to_monitor" ":" <string> ]
  "}"
```

**Constraints:**
- `frequency`: Must be > 0 (validation every N steps)
- `metric_to_monitor`: Must be a metric defined in METRICS block
- `save_best_model`: If true, saves model when monitored metric improves

**Example:**
```okt
VALIDATE {
    on_train: false
    on_validation: true
    frequency: 1
    save_best_model: true
    metric_to_monitor: "loss"
}
```

---

## INFERENCE Block

The `INFERENCE` block defines how the model behaves during inference, prediction, or interactive chat.

```ebnf
<inference_block> ::= 
  "INFERENCE" "{"
      <inference_mode>
      [<inference_format>]
      [<inference_exit_command>]
      [<inference_params>]
      [<inference_control>]
  "}"

<inference_mode> ::=
  "mode" ":" ("chat" | "intent" | "translate" | "classify" | "custom")

<inference_format> ::=
  "format" ":" <string>

<inference_exit_command> ::=
  "exit_command" ":" <string>

<inference_params> ::=
  "params" "{"
      [<inference_max_length>]
      [<inference_temperature>]
      [<inference_top_p>]
      [<inference_beams>]
      [<inference_do_sample>]
      [<inference_top_k>]
      [<inference_repetition_penalty>]
  "}"

<inference_max_length> ::=
  "max_length" ":" <number>

<inference_temperature> ::=
  "temperature" ":" <decimal>

<inference_top_p> ::=
  "top_p" ":" <decimal>

<inference_beams> ::=
  "beams" ":" <number>

<inference_do_sample> ::=
  "do_sample" ":" <boolean>

<inference_top_k> ::=
  "top_k" ":" <number>

<inference_repetition_penalty> ::=
  "repetition_penalty" ":" <decimal>

<inference_control> ::=
  "CONTROL" "{"
      { <control_if> | <control_when> | <control_every> | <control_set> | <control_stop> | <control_log> | <control_save> | <control_retry> | <control_regenerate> }
  "}"
```

**Supported format patterns:**

The `format` field supports template strings with variables:

| Use case | Example |
|----------|---------|
| Chat | `"User: {input}\nAssistant:"` |
| Free | `"{input}"` |
| Translation | `"translate English to Portuguese: {input}"` |
| Intent | `"{input}"` |
| QA/RAG | `"Context: {context}\nQuestion: {input}\nAnswer:"` |
| LLaMA style | `"<|user|>\n{input}\n<|assistant|>\n"` |

**Supported variables:**
- `{input}` → user input
- `{context}` → optional context (for RAG/QA)
- `{labels}` → class list for classification

**Constraints:**
- `mode`: Defines the inference behavior type
- `format`: Template string with variable placeholders
- `max_length`: Must be > 0 and <= 8192
- `temperature`: Must be >= 0.0 and <= 2.0
- `top_p`: Must be > 0.0 and <= 1.0
- `top_k`: Must be >= 0 (0 = disabled)
- `beams`: Must be >= 1 (for beam search)
- `do_sample`: Boolean (true/false)
- `repetition_penalty`: Must be > 0.0 and <= 2.0

**Example (Chat mode):**
```okt
INFERENCE {
    mode: "chat"
    format: "User: {input}\nAssistant:"
    exit_command: "/exit"
    
    params {
        max_length: 120
        temperature: 0.7
        top_p: 0.9
        beams: 2
        do_sample: true
    }
    
    CONTROL {
        IF confidence < 0.3 { RETRY }
        IF repetition > 3 { REGENERATE }
        IF hallucination_score > 0.5 { REPLACE WITH "Desculpe, não tenho certeza." }
    }
}
```

**Example (Translation mode):**
```okt
INFERENCE {
    mode: "translate"
    format: "translate English to Portuguese: {input}"
    
    params {
        max_length: 200
        temperature: 0.5
        top_p: 0.95
    }
}
```

**Example (Classification mode):**
```okt
INFERENCE {
    mode: "classify"
    format: "{input}"
    
    params {
        temperature: 0.1
        top_k: 5
    }
}
```

---

## CONTROL Block — Decision Engine

The `CONTROL` block enables logical, conditional, event-based, and metric-based decisions during training and inference. It introduces a cognitive-level abstraction that allows AI models to take decisions, self-adjust, and self-regulate in a declarative and clean way.

```ebnf
<control_block> ::=
  "CONTROL" "{"
      { <control_event> | <control_if> | <control_when> | <control_every> | <control_set> | <control_stop> | <control_log> | <control_save> | <control_retry> | <control_regenerate> | <control_stop_training> | <control_decrease> | <control_increase> }
  "}"

<control_event> ::=
  <on_step_end> | <on_epoch_end> | <on_memory_low> | <on_nan> | <on_plateau> | <validate_every>

<on_step_end> ::=
  "on_step_end" "{"
      { <control_log> | <control_save> | <control_set> | <control_if> | <control_when> | <control_every> }
  "}"

<on_epoch_end> ::=
  "on_epoch_end" "{"
      { <control_log> | <control_save> | <control_set> | <control_stop_training> | <control_if> | <control_when> | <control_every> }
  "}"

<on_memory_low> ::=
  "on_memory_low" "{"
      { <control_set> | <control_stop> | <control_if> | <control_when> }
  "}"

<on_nan> ::=
  "on_nan" "{"
      { <control_stop_training> | <control_log> | <control_if> }
  "}"

<on_plateau> ::=
  "on_plateau" "{"
      { <control_decrease> | <control_increase> | <control_set> | <control_if> | <control_when> }
  "}"

<validate_every> ::=
  "validate_every" ":" <number>

<control_if> ::=
  "IF" <condition> "{"
      { <control_set> | <control_stop> | <control_log> | <control_save> | <control_stop_training> | <control_decrease> | <control_increase> | <control_retry> | <control_regenerate> | <control_if> | <control_when> | <control_every> }
  "}"

<control_when> ::=
  "WHEN" <condition> "{"
      { <control_set> | <control_stop> | <control_log> | <control_if> | <control_when> }
  "}"

<control_every> ::=
  "EVERY" <number> ("steps" | "epochs") "{"
      { <control_save> | <control_log> | <control_set> | <control_if> | <control_when> }
  "}"

<control_set> ::=
  "SET" <identifier> "=" <value>

<control_stop> ::=
  "STOP"

<control_log> ::=
  "LOG" ( <metric_name> | <string> )

<control_save> ::=
  "SAVE" ( "model" | "checkpoint" | <string> )

<control_retry> ::=
  "RETRY"

<control_regenerate> ::=
  "REGENERATE"

<control_stop_training> ::=
  "STOP_TRAINING"

<control_decrease> ::=
  "DECREASE" <identifier> "BY" <number>

<control_increase> ::=
  "INCREASE" <identifier> "BY" <number>

<condition> ::=
  <metric_name> <comparison_operator> <value>

<comparison_operator> ::=
  ">" | "<" | ">=" | "<=" | "==" | "!="

<value> ::=
  <number> | <decimal> | <string> | <boolean> | <identifier>

<metric_name> ::=
  "loss" | "val_loss" | "accuracy" | "val_accuracy" | "gpu_memory" | "ram_usage" | "confidence" | "hallucination_score" | <custom_metric>

<identifier> ::=
  "LR" | "learning_rate" | "batch_size" | "temperature" | <custom_identifier>
```

**Supported events/hooks:**

| Event | Description |
|-------|-------------|
| `on_step_end` | Executed at the end of each training step |
| `on_epoch_end` | Executed at the end of each epoch |
| `validate_every` | Execute validation every X steps |
| `on_memory_low` | Triggered when GPU/RAM is low |
| `on_nan` | Triggered when NaN values are detected |
| `on_plateau` | Triggered when loss is stagnant (plateau) |

**Supported directives:**

- `IF` - Conditional logic based on metrics
- `WHEN` - Event-based conditional logic
- `EVERY` - Periodic actions (every N steps)
- `SET` - Set parameter values dynamically
- `STOP` - Stop current operation
- `LOG` - Log metrics or messages
- `SAVE` - Save model or checkpoint
- `RETRY` - Retry inference generation
- `REGENERATE` - Regenerate output
- `STOP_TRAINING` - Stop training process
- `DECREASE` - Decrease parameter value
- `INCREASE` - Increase parameter value

**Nested Blocks Support:**

The CONTROL block in OktoScript supports nested logic, event-driven triggers, and conditional reasoning. You can nest IF / WHEN / EVERY statements inside lifecycle hooks like `on_step_end` and `on_epoch_end`, allowing dynamic, real-time decision making during training or inference.

**Example (Basic):**
```okt
CONTROL {
    on_step_end {
        LOG loss
    }
    
    on_epoch_end {
        SAVE model
        LOG "Epoch completed"
    }
    
    validate_every: 200
    
    IF loss > 2.0 {
        SET LR = 0.0001
        LOG "High loss detected"
    }
    
    IF val_loss > 2.5 {
        STOP_TRAINING
    }
    
    WHEN gpu_memory < 16GB {
        SET batch_size = 4
    }
    
    EVERY 500 steps {
        SAVE checkpoint
    }
    
    IF accuracy < 0.4 {
        DECREASE LR BY 0.5
    }
}
```

**Example (Nested Blocks in Events):**
```okt
CONTROL {
    on_epoch_end {
        IF loss > 2.0 {
            SET LR = 0.0001
            LOG "High loss detected"
        }
        
        IF val_loss > 2.5 {
            STOP_TRAINING
        }
        
        IF accuracy > 0.9 {
            SAVE "best_model"
            LOG "High accuracy reached"
        }
    }
}
```

**Example (Advanced Nested Logic):**
```okt
CONTROL {
    on_epoch_end {
        EVERY 2 epochs {
            SAVE "checkpoint_epoch_{epoch}"
        }
        
        IF loss > 2.0 {
            SET LR = 0.00005
            LOG "Loss still high after epoch"
            
            WHEN gpu_usage > 90% {
                SET batch_size = 2
                LOG "Reducing batch size due to GPU pressure"
            }
            
            IF val_loss > 3.0 {
                STOP_TRAINING
            }
        }
    }
}
```

**Example (Context-Based Control):**
```okt
CONTROL {
    IF epoch == 1 {
        LOG "Warmup stage"
    }
    
    IF epoch > 5 AND accuracy < 0.6 {
        SET LR = 0.00001
        LOG "Model is stagnated"
    }
    
    IF epoch > 10 AND loss > 1.8 {
        STOP_TRAINING
    }
}
```

**Example (Inference CONTROL):**
```okt
INFERENCE {
    mode: "chat"
    format: "User: {input}\nAssistant:"
    
    CONTROL {
        IF confidence < 0.3 { RETRY }
        IF repetition > 3 { REGENERATE }
        IF toxic == true { REPLACE WITH "Not allowed" }
    }
}
```

**Example (Intent Classification CONTROL):**
```okt
INFERENCE {
    mode: "intent-classification"
    labels: ["greeting", "order", "complaint", "bye"]
}

CONTROL {
    IF label == "complaint" {
        RETURN "I'm sorry to hear that. How can I help?"
    }
    
    IF confidence < 0.4 {
        RETURN "Could you please repeat?"
    }
}
```

**Note:** OktoScript enables true declarative AI governance. CONTROL blocks can contain nested conditions and nested event triggers, making it a unique declarative decision-making language in the market.

**Philosophy:**

OktoScript keeps the surface clean and simple, while the engine behind it performs complex cognitive decision-making.

- **CONTROL** defines logic
- **MONITOR** defines awareness
- **GUARD** defines safety
- **BEHAVIOR** defines personality
- **EXPLORER** defines optimization
- **STABILITY** defines reliability

Simple to read. Powerful to execute.

---

## EXPORT Block

```ebnf
<export_block> ::= 
  "EXPORT" "{"
      "format" ":" "[" <export_format_list> "]"
      "path" ":" <path>
      [ "quantization" ":" ("int8" | "int4" | "fp16" | "fp32") ]
      [ "optimize_for" ":" ("speed" | "size" | "accuracy") ]
  "}"

<export_format_list> ::= 
  "gguf" |
  "onnx" |
  "okm" |
  "safetensors" |
  "tflite"
```

**Format-specific constraints:**
- `gguf`: Requires quantization (int8, int4, or fp16)
- `onnx`: Best for production deployment
- `okm`: OktoSeek optimized format (requires OktoSeek SDK)
- `safetensors`: Standard PyTorch format
- `tflite`: For mobile deployment (Android/iOS)

**Example:**
```okt
EXPORT {
    format: ["gguf", "onnx", "okm", "safetensors"]
    path: "export/"
    quantization: "int8"
    optimize_for: "speed"
}
```

---

## DEPLOY Block

The `DEPLOY` block defines deployment configuration for the model. The engine will create the server, generate routes, export in the required format, and configure limits and authentication.

```ebnf
<deploy_block> ::=
  "DEPLOY" "{"
      "target" ":" ("local" | "cloud" | "edge" | "api" | "android" | "ios" | "web" | "desktop")
      [ "endpoint" ":" <string> ]
      [ "host" ":" <string> ]
      [ "requires_auth" ":" ("true" | "false") ]
      [ "port" ":" <number> ]
      [ "max_concurrent_requests" ":" <number> ]
      [ "protocol" ":" ("http" | "https" | "grpc" | "ws") ]
      [ "format" ":" ("onnx" | "tflite" | "gguf" | "pt" | "okm") ]
  "}"
```

**Target-specific requirements:**
- `api`: Requires `endpoint`, `host`, and `port`
- `android`, `ios`: Requires `.okm` or `.tflite` format
- `web`: Requires ONNX format
- `edge`: Requires quantized model (int8 or int4)

**Protocol options:**
- `http` - HTTP REST API
- `https` - HTTPS REST API
- `grpc` - gRPC protocol
- `ws` - WebSocket protocol

**Format options:**
- `onnx` - ONNX format (production-ready)
- `tflite` - TensorFlow Lite (mobile)
- `gguf` - GGUF format (local inference)
- `pt` - PyTorch format
- `okm` - OktoModel format (OktoSeek optimized)

**Example (API Deployment):**
```okt
DEPLOY {
    target: "api"
    host: "0.0.0.0"
    endpoint: "/pizzabot"
    requires_auth: true
    port: 9000
    max_concurrent_requests: 100
    protocol: "http"
    format: "onnx"
}
```

**Example (Mobile Deployment):**
```okt
DEPLOY {
    target: "android"
    format: "tflite"
}
```

---

## SECURITY Block

The `SECURITY` block defines security measures for input validation, output validation, rate limiting, and encryption.

```ebnf
<security_block> ::=
  "SECURITY" "{"
      [ <input_validation> ]
      [ <output_validation> ]
      [ <rate_limit> ]
      [ <encryption> ]
  "}"

<input_validation> ::=
  "input_validation" "{"
      [ "max_length" ":" <number> ]
      [ "disallow_patterns" ":" "[" <string_list> "]" ]
  "}"

<output_validation> ::=
  "output_validation" "{"
      [ "prevent_data_leak" ":" ("true" | "false") ]
      [ "mask_personal_info" ":" ("true" | "false") ]
  "}"

<rate_limit> ::=
  "rate_limit" "{"
      [ "max_requests_per_minute" ":" <number> ]
  "}"

<encryption> ::=
  "encryption" "{"
      [ "algorithm" ":" ("AES-256" | "SHA-256" | "RSA") ]
  "}"
```

**Input validation:**
- `max_length` - Maximum input length in characters
- `disallow_patterns` - List of patterns to block (e.g., SQL injection, XSS)

**Output validation:**
- `prevent_data_leak` - Prevent training data from appearing in outputs
- `mask_personal_info` - Mask personal information in outputs

**Rate limiting:**
- `max_requests_per_minute` - Maximum requests per minute per client

**Encryption:**
- `AES-256` - AES-256 encryption
- `SHA-256` - SHA-256 hashing
- `RSA` - RSA encryption

**Example:**
```okt
SECURITY {
    input_validation {
        max_length: 500
        disallow_patterns: [
            "<script>",
            "DROP TABLE",
            "rm -rf",
            "sudo",
            "passwd"
        ]
    }
    
    output_validation {
        prevent_data_leak: true
        mask_personal_info: true
    }
    
    rate_limit {
        max_requests_per_minute: 60
    }
    
    encryption {
        algorithm: "AES-256"
    }
}
```

---

## MONITOR Block — Full Metrics Support

The `MONITOR` block tracks ANY available training or system metric. It supports all native and custom metrics, including but not limited to:

**Training Metrics:**
- `loss`, `val_loss` - Training and validation loss
- `accuracy`, `val_accuracy` - Classification accuracy
- `precision`, `recall`, `f1_score` - Classification metrics
- `perplexity` - Language model perplexity
- `bleu`, `rouge` - Generation quality metrics
- `cer`, `wer` - Character/Word error rates
- `confidence` - Model confidence scores
- `hallucination_score` - Hallucination detection

**System Metrics:**
- `gpu_usage`, `gpu_memory_used`, `gpu_memory_free` - GPU utilization
- `ram_usage`, `cpu_usage` - System resource usage
- `gpu_temperature` - GPU temperature monitoring
- `step_time`, `throughput`, `latency` - Performance metrics
- `token_count` - Token usage statistics

```ebnf
<monitor_block> ::=
  "MONITOR" "{"
      [<monitor_metrics>]
      [<monitor_notify_if>]
      [<monitor_log_to>]
      [<monitor_level>]
      [<monitor_log_system>]
      [<monitor_log_speed>]
      [<monitor_refresh_interval>]
      [<monitor_export_to>]
      [<monitor_dashboard>]
  "}"

<monitor_metrics> ::=
  "metrics" ":" "[" <metric_list> "]"

<monitor_notify_if> ::=
  "notify_if" "{"
      { <notify_condition> }
  "}"

<notify_condition> ::=
  <metric_name> <comparison_operator> <value>

<monitor_log_to> ::=
  "log_to" ":" <path>

<monitor_level> ::=
  "level" ":" ("basic" | "full")

<monitor_log_system> ::=
  "log_system" ":" "[" <system_metric_list> "]"

<monitor_log_speed> ::=
  "log_speed" ":" "[" <speed_metric_list> "]"

<monitor_refresh_interval> ::=
  "refresh_interval" ":" <time_interval>

<monitor_export_to> ::=
  "export_to" ":" <path>

<monitor_dashboard> ::=
  "dashboard" ":" <boolean>

<metric_list> ::=
  <string> { "," <string> }

<system_metric_list> ::=
  ("gpu_memory_used" | "gpu_memory_free" | "gpu_usage" | "cpu_usage" | "ram_usage" | "ram_used" | "disk_io" | "gpu_temperature" | "temperature") { "," <system_metric> }

<speed_metric_list> ::=
  ("tokens_per_second" | "samples_per_second" | "throughput" | "latency" | "step_time") { "," <speed_metric> }

<time_interval> ::=
  <number> ("s" | "ms")
```

**Supported Metrics (Complete List):**

**Training Metrics:**
- `loss`, `val_loss` - Loss values
- `accuracy`, `val_accuracy` - Accuracy metrics
- `precision`, `recall`, `f1_score` - Classification metrics
- `perplexity` - Language model perplexity
- `bleu`, `rouge`, `rouge_l`, `rouge_1`, `rouge_2` - Generation metrics
- `cer`, `wer` - Error rates
- `confidence` - Confidence scores
- `hallucination_score` - Hallucination detection

**System Metrics:**
- `gpu_usage`, `gpu_memory_used`, `gpu_memory_free`, `gpu_temperature` - GPU metrics
- `ram_usage`, `cpu_usage` - System resources
- `step_time`, `throughput`, `latency` - Performance
- `token_count` - Token statistics

**Constraints:**
- `metrics`: Array of metric names to track
- `notify_if`: Conditions that trigger notifications
- `log_to`: Path to log file (optional)
- GPU metrics only logged if CUDA is available
- `refresh_interval` must be >= 1s

**Example:**
```okt
MONITOR {
    metrics: [
        "loss",
        "accuracy",
        "val_loss",
        "gpu_usage",
        "ram_usage",
        "throughput",
        "latency",
        "confidence"
    ]
    
    notify_if {
        loss > 2.0
        gpu_usage > 90%
        temperature > 85
        hallucination_score > 0.5
    }
    
    log_to: "logs/training.log"
}
```

**Example (Full monitoring):**
```okt
MONITOR {
    level: "full"
    metrics: [
        "loss",
        "val_loss",
        "accuracy",
        "f1",
        "perplexity",
        "confidence",
        "hallucination_score"
    ]
    log_system: [
        "gpu_memory_used",
        "gpu_memory_free",
        "cpu_usage",
        "ram_used",
        "gpu_temperature"
    ]
    log_speed: [
        "tokens_per_second",
        "samples_per_second",
        "throughput",
        "latency"
    ]
    notify_if {
        loss > 2.0
        gpu_usage > 90%
        val_loss > 2.5
    }
    refresh_interval: 2s
    export_to: "runs/logs/system.json"
    dashboard: true
    log_to: "logs/training.log"
}
```

**Integration with METRICS and LOGGING:**
- `MONITOR` extends (does not replace) `METRICS` and `LOGGING`
- System metrics are logged separately from training metrics
- Dashboard provides real-time visualization (if `dashboard: true`)
- `notify_if` triggers alerts when conditions are met

---

## GUARD Block — Safety / Ethics / Protection

The `GUARD` block defines safety rules during generation and training. It prevents harmful outputs and ensures ethical AI behavior. The engine knows exactly what to prevent, how to detect violations, and what action to take.

```ebnf
<guard_block> ::=
  "GUARD" "{"
      [<guard_prevent>]
      [<guard_detect_using>]
      [<guard_on_violation>]
  "}"

<guard_prevent> ::=
  "prevent" "{"
      { <prevention_type> }
  "}"

<guard_detect_using> ::=
  "detect_using" ":" "[" ("classifier" | "embedding" | "regex" | "rule_engine" | "ml_model") { "," ("classifier" | "embedding" | "regex" | "rule_engine" | "ml_model") } "]"

<prevention_type> ::=
  "hallucination" |
  "toxicity" |
  "bias" |
  "data_leak" |
  "unsafe_code" |
  "personal_data" |
  "illegal_content"

<guard_on_violation> ::=
  "on_violation" "{"
      <violation_action>
      [ "with_message" ":" <string> ]
  "}"

<violation_action> ::=
  "STOP" | "ALERT" | "REPLACE" | "LOG"
```

**Prevention types:**
- `hallucination` - Prevents fabricated or false information
- `toxicity` - Prevents toxic, harmful, or offensive content
- `bias` - Prevents biased or discriminatory outputs
- `data_leak` - Prevents training data leakage
- `unsafe_code` - Prevents unsafe code generation
- `personal_data` - Prevents personal information leakage
- `illegal_content` - Prevents illegal content generation

**Detection methods:**
- `classifier` - Use ML classifier to detect violations
- `embedding` - Use embedding similarity to detect violations
- `regex` - Use regex patterns to detect violations
- `rule_engine` - Use rule-based engine to detect violations
- `ml_model` - Use custom ML model to detect violations

**Violation actions:**
- `STOP` - Stop generation immediately
- `ALERT` - Log alert and continue
- `REPLACE` - Replace with safe fallback (requires `with_message`)
- `LOG` - Log violation for analysis

**Example (Strict Mode):**
```okt
GUARD {
    prevent {
        hallucination
        toxicity
        bias
        data_leak
        illegal_content
    }
    
    detect_using: ["classifier", "regex", "embedding"]
    
    on_violation {
        REPLACE
        with_message: "Sorry, this request is not allowed."
    }
}
```

**Example (Alert mode):**
```okt
GUARD {
    prevent {
        toxicity
        bias
    }
    
    detect_using: ["classifier", "rule_engine"]
    
    on_violation {
        ALERT
    }
}
```

---

## BEHAVIOR Block — Model Personality

The `BEHAVIOR` block defines how the model should behave in chat/inference. It sets personality, verbosity, language, and content restrictions.

```ebnf
<behavior_block> ::=
  "BEHAVIOR" "{"
      [<behavior_mode>]
      [<behavior_personality>]
      [<behavior_verbosity>]
      [<behavior_language>]
      [<behavior_avoid>]
      [<behavior_fallback>]
      [<behavior_style_prompt>]
  "}"

<behavior_mode> ::=
  "mode" ":" ("chat" | "completion" | "instruction" | "classifier")

<behavior_personality> ::=
  "personality" ":" ("professional" | "friendly" | "assistant" | "casual" | "formal" | "creative")

<behavior_verbosity> ::=
  "verbosity" ":" ("low" | "medium" | "high")

<behavior_language> ::=
  "language" ":" ("en" | "pt-BR" | "es" | "fr" | "de" | "it" | "ja" | "zh" | "multilingual")

<behavior_avoid> ::=
  "avoid" ":" "[" <string_list> "]"

<behavior_fallback> ::=
  "fallback" ":" <string>

<behavior_style_prompt> ::=
  "prompt_style" ":" <string>
```

**Mode types:**
- `chat` - Conversational chat mode
- `completion` - Text completion mode
- `instruction` - Instruction-following mode
- `classifier` - Classification mode

**Personality types:**
- `professional` - Formal, business-like responses
- `friendly` - Warm, approachable tone
- `assistant` - Helpful, service-oriented
- `casual` - Relaxed, informal tone
- `formal` - Very formal, academic tone
- `creative` - Imaginative, expressive responses

**Verbosity levels:**
- `low` - Concise, brief responses
- `medium` - Balanced detail
- `high` - Detailed, comprehensive responses

**prompt_style allows you to define:**
- ChatGPT-style format
- Translation format
- Classification format
- Custom format (e.g., NLP tasks)

**Example (Professional Chatbot):**
```okt
BEHAVIOR {
    mode: "chat"
    personality: "professional"
    verbosity: "medium"
    language: "pt-BR"
    avoid: ["violence", "hate", "politics"]
    fallback: "Como posso ajudar?"
    prompt_style: "User: {input}\nAssistant:"
}
```

**Example (Friendly assistant):**
```okt
BEHAVIOR {
    mode: "chat"
    personality: "friendly"
    verbosity: "high"
    language: "en"
    avoid: ["violence", "explicit content"]
    fallback: "I'm here to help! How can I assist you?"
    prompt_style: "User: {input}\nAssistant:"
}
```

---

## EXPLORER Block — Parameter Search

The `EXPLORER` block enables basic hyperparameter exploration (AutoML-style). It automatically tests different parameter combinations and selects the best configuration.

```ebnf
<explorer_block> ::=
  "EXPLORER" "{"
      <explorer_try>
      [<explorer_max_tests>]
      [<explorer_pick_best_by>]
  "}"

<explorer_try> ::=
  "try" "{"
      { <explorer_lr> | <explorer_batch_size> | <explorer_optimizer> | <explorer_scheduler> | <explorer_other> }
  "}"

<explorer_lr> ::=
  "lr" ":" "[" <decimal_list> "]"

<explorer_batch_size> ::=
  "batch_size" ":" "[" <number_list> "]"

<explorer_optimizer> ::=
  "optimizer" ":" "[" <string_list> "]"

<explorer_scheduler> ::=
  "scheduler" ":" "[" <string_list> "]"

<explorer_other> ::=
  <identifier> ":" "[" <value_list> "]"

<explorer_max_tests> ::=
  "max_tests" ":" <number>

<explorer_pick_best_by> ::=
  "pick_best_by" ":" <metric_name>

<decimal_list> ::=
  <decimal> { "," <decimal> }

<number_list> ::=
  <number> { "," <number> }

<value_list> ::=
  <value> { "," <value> }
```

**Constraints:**
- `max_tests`: Must be <= 50 (to prevent excessive exploration)
- `pick_best_by`: Must be a valid metric name (e.g., `"val_loss"`, `"accuracy"`)
- Explorer will test combinations and select the best configuration based on the specified metric

**Example:**
```okt
EXPLORER {
    try {
        lr: [0.001, 0.0005, 0.0001]
        batch_size: [4, 8, 16]
        optimizer: ["adamw", "sgd"]
    }
    
    max_tests: 5
    pick_best_by: "val_loss"
}
```

**Example (Extended exploration):**
```okt
EXPLORER {
    try {
        lr: [0.0003, 0.0001, 0.00005]
        batch_size: [8, 16, 32]
        optimizer: ["adamw", "adam"]
        scheduler: ["cosine", "linear"]
    }
    
    max_tests: 12
    pick_best_by: "val_accuracy"
}
```

---

## STABILITY Block — Training Safety

The `STABILITY` block controls training stability and prevents common training failures.

```ebnf
<stability_block> ::=
  "STABILITY" "{"
      [<stability_stop_if_nan>]
      [<stability_stop_if_diverges>]
      [<stability_min_improvement>]
  "}"

<stability_stop_if_nan> ::=
  "stop_if_nan" ":" <boolean>

<stability_stop_if_diverges> ::=
  "stop_if_diverges" ":" <boolean>

<stability_min_improvement> ::=
  "min_improvement" ":" <decimal>
```

**Constraints:**
- `stop_if_nan`: Boolean - Stop training if NaN values are detected
- `stop_if_diverges`: Boolean - Stop training if loss diverges
- `min_improvement`: Float - Minimum improvement threshold (e.g., 0.001)

**Example:**
```okt
STABILITY {
    stop_if_nan: true
    stop_if_diverges: true
    min_improvement: 0.001
}
```

**Example (Relaxed stability):**
```okt
STABILITY {
    stop_if_nan: true
    stop_if_diverges: false
    min_improvement: 0.0001
}
```

---

## Boolean Support

The OktoScript language supports boolean values:

- `true`
- `false`

**Supported in:**
- `CONTROL` block conditions and actions
- `STABILITY` block flags
- `BEHAVIOR` block settings
- `GUARD` block actions
- `MONITOR` block dashboard flag
- `INFERENCE` block `do_sample` parameter
- Any block that requires boolean values

**Example:**
```okt
STABILITY {
    stop_if_nan: true
    stop_if_diverges: false
}

BEHAVIOR {
    personality: "friendly"
}

MONITOR {
    dashboard: true
}

INFERENCE {
    params {
        do_sample: true
    }
}
```

---

## LOGGING Block

```ebnf
<logging_block> ::= 
  "LOGGING" "{"
      "save_logs" ":" ("true" | "false")
      "metrics_file" ":" <path>
      "training_file" ":" <path>
      [ "log_level" ":" ("debug" | "info" | "warning" | "error") ]
      [ "log_every" ":" <number> ]
  "}"
```

**Example:**
```okt
LOGGING {
    save_logs: true
    metrics_file: "runs/pizzabot-v1/metrics.json"
    training_file: "runs/pizzabot-v1/training_logs.json"
    log_level: "info"
    log_every: 10
}
```

---

## Model Inheritance

OktoScript supports model inheritance to reduce code duplication and enable configuration reuse.

**Syntax:**
```okt
# Base model definition (named)
MODEL "base-transformer" {
    architecture: "transformer"
    context_window: 2048
    precision: "fp16"
}

# Child model inheriting from base
MODEL {
    inherit: "base-transformer"
    base: "oktoseek/custom-model"
    parameters: 250M
    # Overrides: precision stays "fp16" from parent
    # New: base and parameters are set
}
```

**Inheritance rules:**
1. Child model inherits all fields from parent
2. Child can override any inherited field
3. Fields not specified in child use parent values
4. Inheritance chain can be multiple levels (parent → child → grandchild)
5. Circular inheritance is not allowed

**Example with multiple inheritance:**
```okt
# Grandparent
MODEL "base-config" {
    architecture: "transformer"
    precision: "fp16"
}

# Parent
MODEL "medium-model" {
    inherit: "base-config"
    parameters: 120M
    context_window: 2048
}

# Child
MODEL {
    inherit: "medium-model"
    base: "oktoseek/specialized-model"
    parameters: 250M
}
```

---

## Extension Points & Hooks

OktoScript supports extension points for custom logic integration.

### HOOKS Block

```ebnf
<hooks_block> ::=
  "HOOKS" "{"
      [ "before_train" ":" <script_path> ]
      [ "after_train" ":" <script_path> ]
      [ "before_epoch" ":" <script_path> ]
      [ "after_epoch" ":" <script_path> ]
      [ "on_checkpoint" ":" <script_path> ]
      [ "custom_metric" ":" <script_path> ]
  "}"
```

**Hook script format:**
- Python scripts (`.py`) - Most common
- JavaScript/Node.js (`.js`) - For web integrations
- Shell scripts (`.sh`) - For system operations

**Hook script interface:**
```python
# before_train.py
def before_train(config, dataset, model):
    # Custom preprocessing
    # Modify config if needed
    return config

# after_epoch.py
def after_epoch(epoch, metrics, model_state):
    # Custom logging, early stopping logic
    # Return True to stop training
    return False
```

**Example:**
```okt
HOOKS {
    before_train: "scripts/preprocess.py"
    after_epoch: "scripts/custom_early_stop.py"
    on_checkpoint: "scripts/backup_checkpoint.sh"
    custom_metric: "scripts/toxicity_calculator.py"
}
```

### Python Integration

OktoScript can call Python functions directly:

```okt
HOOKS {
    before_train: "python:my_module.preprocess_data"
    custom_metric: "python:metrics.custom_f1_score"
}
```

### API Integration

```okt
HOOKS {
    after_train: "api:https://api.example.com/log_training"
    on_checkpoint: "api:https://api.example.com/upload_checkpoint"
}
```

---

## Validation Rules

### File Structure Validation

1. **Required files:**
   - `okt.yaml` must exist in project root
   - Dataset files specified in DATASET block must exist
   - Model base path must be valid (if local path)

2. **Field validation:**
   - All required fields must be present
   - Field types must match grammar specification
   - Numeric values must be within allowed ranges
   - String values must match allowed patterns

3. **Dependency validation:**
   - If `inherit` is used, parent model must exist
   - If `resume_from_checkpoint` is used, checkpoint must exist
   - Export formats must be compatible with model architecture

### Runtime Validation

**Dataset validation:**
- File exists and is readable
- Format matches declared format
- Required columns/fields present (for structured data)
- File size within limits (max 10GB per file)

**Model validation:**
- Base model exists (if local) or is downloadable (if remote)
- Model architecture compatible with dataset type
- Model size fits available memory

**Training validation:**
- Device available (GPU if specified)
- Sufficient disk space for checkpoints
- Batch size fits in memory

### Error Messages

Common validation errors and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `Dataset file not found` | Path in DATASET block doesn't exist | Check file path, use absolute or relative path |
| `Invalid optimizer: 'invalid'` | Optimizer not in allowed list | Use one of: adam, adamw, sgd, rmsprop, adafactor, lamb |
| `Model base not found` | Base model path invalid | Verify model path or HuggingFace model name |
| `Checkpoint not found` | Resume checkpoint doesn't exist | Check checkpoint path or remove resume_from_checkpoint |
| `Insufficient memory` | Batch size too large | Reduce batch_size or enable gradient_accumulation |
| `Invalid metric for task` | Metric incompatible with task type | Use appropriate metrics (e.g., accuracy for classification) |

---

## Troubleshooting

### Common Issues

**1. Training fails with "Out of Memory"**

**Symptoms:**
- CUDA out of memory error
- Training crashes after a few steps

**Solutions:**
```okt
TRAIN {
    batch_size: 8  # Reduce from 32
    gradient_accumulation: 4  # Increase to maintain effective batch size
    precision: "fp16"  # Use mixed precision
}
```

**2. Model not improving (loss plateau)**

**Symptoms:**
- Loss stops decreasing
- Metrics remain constant

**Solutions:**
```okt
TRAIN {
    learning_rate: 0.0001  # Try different learning rate
    scheduler: "cosine_with_restarts"  # Use learning rate schedule
    weight_decay: 0.01  # Add regularization
}

VALIDATE {
    save_best_model: true
    metric_to_monitor: "loss"
}
```

**3. Dataset format errors**

**Symptoms:**
- "Invalid dataset format"
- "Missing required columns"

**Solutions:**
- Verify dataset format matches declared format
- For JSONL: Ensure each line is valid JSON with required fields
- For CSV: Check column names match expected schema
- Use `okto validate` to check dataset before training

**4. Export fails**

**Symptoms:**
- "Export format not supported"
- "Quantization failed"

**Solutions:**
- Ensure model architecture supports export format
- For GGUF: Model must be quantized (use int8 or int4)
- For ONNX: Model must be ONNX-compatible architecture
- Check available disk space

**5. Inference produces poor results**

**Symptoms:**
- Low quality outputs
- Repetitive text
- Off-topic responses

**Solutions:**
```okt
INFERENCE {
    temperature: 0.7  # Lower = more deterministic
    top_p: 0.9  # Nucleus sampling
    top_k: 40  # Limit vocabulary
    repetition_penalty: 1.2  # Reduce repetition
    max_tokens: 200  # Limit length
}
```

### Debug Mode

Enable debug logging:

```okt
LOGGING {
    save_logs: true
    log_level: "debug"
    log_every: 1
}
```

Run with verbose output:
```bash
okto run project.okt --verbose --debug
```

### Performance Optimization

**For faster training:**
```okt
TRAIN {
    batch_size: 64  # Larger batches
    gradient_accumulation: 1  # No accumulation
    mixed_precision: true  # FP16
    device: "cuda"
}
```

**For memory efficiency:**
```okt
TRAIN {
    batch_size: 4
    gradient_accumulation: 8
    precision: "fp16"
    checkpoint_steps: 50  # Save more frequently
}
```

---

## Terminal / Basic Types

```ebnf
<string> ::= '"' { any-character-except-quote } '"'

<string_list> ::= <string> { "," <string> }

<path> ::= '"' { any-character-except-quote } '"'

<number> ::= digit { digit }

<decimal> ::= digit { digit } "." digit { digit }

<boolean> ::= "true" | "false"

digit ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
```

**Type constraints:**
- `string`: UTF-8 encoded, max 10,000 characters
- `path`: Can be absolute or relative, must be valid filesystem path
- `number`: Integer, range depends on field (typically 0 to 2^31-1)
- `decimal`: Floating point, precision up to 6 decimal places
- `boolean`: Literal values `true` or `false` (lowercase)

---

## Full Script Example

Complete example demonstrating all new blocks:

```okt
# okto_version: "1.2"
PROJECT "oktogpt"
DESCRIPTION "Complete example with all new blocks"

MODEL {
    name: "oktogpt"
    base: "google/flan-t5-base"
    device: "cuda"
    
    ADAPTER {
        type: "lora"
        path: "D:/model_trainee/phase1_sharegpt/ep2"
        rank: 16
        alpha: 32
    }
}

MONITOR {
    metrics: [
        "loss",
        "val_loss",
        "accuracy",
        "gpu_usage",
        "ram_usage",
        "throughput",
        "latency",
        "confidence"
    ]
    
    notify_if {
        loss > 2.0
        gpu_usage > 90%
        temperature > 85
        hallucination_score > 0.5
    }
    
    log_to: "logs/training.log"
}

BEHAVIOR {
    personality: "assistant"
    language: "pt-BR"
    verbosity: "medium"
    avoid: ["politics", "violence", "hate"]
    fallback: "Como posso ajudar?"
}

STABILITY {
    stop_if_nan: true
    stop_if_diverges: true
    min_improvement: 0.001
}

EXPLORER {
    try {
        lr: [0.0003, 0.0001]
        batch_size: [4, 8]
    }
    max_tests: 4
    pick_best_by: "val_loss"
}

CONTROL {
    on_epoch_end {
        SAVE model
        LOG "Epoch completed"
    }
    
    IF val_loss > 2.0 {
        STOP_TRAINING
    }
    
    WHEN gpu_memory < 12GB {
        SET batch_size = 4
    }
}

INFERENCE {
    mode: "chat"
    format: "User: {input}\nAssistant:"
    
    params {
        temperature: 0.7
        max_length: 120
        top_p: 0.9
        beams: 2
        do_sample: true
    }
    
    CONTROL {
        IF confidence < 0.3 { RETRY }
        IF hallucination_score > 0.5 { REPLACE WITH "Desculpe, não tenho certeza." }
    }
    
    exit_command: "/exit"
}

GUARD {
    prevent {
        hallucination
        toxicity
        bias
        data_leak
        unsafe_code
    }
    
    on_violation {
        STOP
    }
}
```

---

## BLAS Block — GPU Acceleration (v1.3+)

The `BLAS` block allows users to configure GPU acceleration settings for training and inference. This is optional and provides fine-grained control over matrix operations.

```ebnf
<blas_block> ::=
  "BLAS" "{"
      [<blas_backend>]
      [<blas_precision>]
      [<blas_streams>]
  "}"

<blas_backend> ::=
  "backend" ":" ("oktoblas" | "cublas" | "auto")

<blas_precision> ::=
  "precision" ":" ("fp16" | "fp32" | "auto")

<blas_streams> ::=
  "streams" ":" <number>
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | enum | `"auto"` | BLAS backend: `"oktoblas"` for OktoBLAS, `"cublas"` for cuBLAS, `"auto"` for automatic |
| `precision` | enum | `"auto"` | Computation precision for matrix operations |
| `streams` | number | `1` | Number of CUDA streams for parallel operations |

**Example:**
```okt
BLAS {
    backend: "oktoblas"
    precision: "fp16"
    streams: 4
}
```

---

## ACCELERATE Block — Operation Optimization (v1.3+)

The `ACCELERATE` block provides hints for which operations should use GPU acceleration.

```ebnf
<accelerate_block> ::=
  "ACCELERATE" "{"
      [<accelerate_gemm>]
      [<accelerate_attention>]
      [<accelerate_fused_ops>]
  "}"

<accelerate_gemm> ::=
  "gemm" ":" ("oktoblas" | "native" | "auto")

<accelerate_attention> ::=
  "attention" ":" ("oktoblas" | "native" | "auto")

<accelerate_fused_ops> ::=
  "fused_ops" ":" <boolean>
```

**Example:**
```okt
ACCELERATE {
    gemm: "oktoblas"
    attention: "oktoblas"
    fused_ops: true
}
```

---

## TENSOR_CORES Block — Tensor Core Control (v1.3+)

The `TENSOR_CORES` block enables explicit control over NVIDIA Tensor Cores usage.

```ebnf
<tensor_cores_block> ::=
  "TENSOR_CORES" "{"
      [<tc_enabled>]
      [<tc_precision>]
  "}"

<tc_enabled> ::=
  "enabled" ":" <boolean>

<tc_precision> ::=
  "precision" ":" ("fp16" | "bf16" | "tf32")
```

**Example:**
```okt
TENSOR_CORES {
    enabled: true
    precision: "fp16"
}
```

---

## Complete Examples

See [`../examples/`](../examples/) for complete working examples:

- [`basic.okt`](../examples/basic.okt) - Minimal example
- [`chatbot.okt`](../examples/chatbot.okt) - Conversational AI
- [`computer_vision.okt`](../examples/computer_vision.okt) - Image classification
- [`recommender.okt`](../examples/recommender.okt) - Recommendation system
- [`pizzabot/`](../examples/pizzabot/) - Complete project example

📊 **Example datasets available in** [`../examples/pizzabot/dataset/`](../examples/pizzabot/dataset/)

---

**Version:** 1.3  
**Last Updated:** December 2025  
**Maintained by:** OktoSeek AI

---

## Version History

### v1.3 (December 2025)
- ✅ Added `BLAS` block for GPU acceleration configuration
- ✅ Added `ACCELERATE` block for operation optimization
- ✅ Added `TENSOR_CORES` block for Tensor Core control
- ✅ OktoBLAS integration for 125% PyTorch FP16 performance
- ✅ 100% backward compatible with v1.0, v1.1, and v1.2

### v1.2 (December 2025)
- ✅ Enhanced `CONTROL` block with nested blocks support
- ✅ Enhanced `BEHAVIOR` block with `mode` and `prompt_style`
- ✅ Enhanced `GUARD` block with `detect_using` and additional prevention types
- ✅ Enhanced `DEPLOY` block with `host`, `protocol`, and `format`
- ✅ Enhanced `SECURITY` block with `input_validation`, `output_validation`, `rate_limit`, and `encryption`
- ✅ Added support for nested IF/WHEN/EVERY statements inside event hooks
- ✅ 100% backward compatible with v1.0 and v1.1

### v1.1 (November 2025)
- ✅ Added `FT_LORA` block for LoRA fine-tuning
- ✅ Added dataset mixing support (`mix_datasets`, `dataset_percent`, `sampling`)
- ✅ Added `MONITOR` block for system telemetry
- ✅ Added version declaration (`# okto_version`)
- ✅ 100% backward compatible with v1.0

### v1.0 (Initial Release)
- Initial OktoScript specification
- Core blocks: PROJECT, DATASET, MODEL, TRAIN, METRICS, EXPORT, DEPLOY
- Model inheritance
- Extension points and hooks

---

## About OktoScript

**OktoScript** is a domain-specific programming language developed by **OktoSeek AI** for building, training, evaluating and exporting AI models. It is part of the OktoSeek ecosystem, which includes OktoSeek IDE, OktoEngine, and various tools for AI development.

### 🌐 OktoScript Web Editor

Try OktoScript online with the **OktoScript Web Editor** at [https://oktoseek.com/editor.php](https://oktoseek.com/editor.php). The editor features:

- **Smart Autocomplete** – Context-aware suggestions based on the current block
- **Real-time Syntax Validation** – Detects errors like nested blocks and missing braces
- **CLI Integration** – Use `okto web` command to open files directly
- **Auto-save to Local** – Saves back to the same location when you load a file

For more information, visit:
- **Official website:** https://www.oktoseek.com
- **Web Editor:** https://oktoseek.com/editor.php
- **GitHub:** https://github.com/oktoseek/oktoscript
- **Hugging Face:** https://huggingface.co/OktoSeek
- **Twitter:** https://x.com/oktoseek
- **YouTube:** https://www.youtube.com/@Oktoseek
