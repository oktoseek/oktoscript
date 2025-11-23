# OktoScript Grammar Specification v1.1

Complete formal grammar for the OktoScript language, developed by **OktoSeek AI**.

> **Version Compatibility:** This specification covers OktoScript v1.1, which is 100% backward compatible with v1.0. Files without version declaration default to v1.0.

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
10. [EXPORT Block](#export-block)
11. [DEPLOY Block](#deploy-block)
12. [SECURITY Block](#security-block)
13. [LOGGING Block](#logging-block)
14. [Model Inheritance](#model-inheritance)
15. [Extension Points & Hooks](#extension-points--hooks)
16. [Validation Rules](#validation-rules)
17. [Troubleshooting](#troubleshooting)
18. [Terminal / Basic Types](#terminal--basic-types)

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
# okto_version: "1.1"
PROJECT "MyModel"
...
```

**Rules:**
- Optional at the top of the file
- If missing, defaults to v1.0
- Must be the first line (comments allowed before)
- Format: `# okto_version: "1.1"` or `# okto_version: "1.0"`

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
      <model_base>
      [<model_architecture>]
      [<model_parameters>]
      [<model_context_window>]
      [<model_precision>]
      [<model_inherit>]
  "}"

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
    base: "oktoseek/pizza-small"
    architecture: "transformer"
    parameters: 120M
    context_window: 2048
    precision: "fp16"
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

```ebnf
<inference_block> ::= 
  "INFERENCE" "{"
      "max_tokens" ":" <number>
      "temperature" ":" <decimal>
      "top_p" ":" <decimal>
      "top_k" ":" <number>
      [ "repetition_penalty" ":" <decimal> ]
      [ "stop_sequences" ":" "[" <string_list> "]" ]
  "}"
```

**Constraints:**
- `max_tokens`: Must be > 0 and <= 8192
- `temperature`: Must be >= 0.0 and <= 2.0
- `top_p`: Must be > 0.0 and <= 1.0
- `top_k`: Must be >= 0 (0 = disabled)
- `repetition_penalty`: Must be > 0.0 and <= 2.0

**Example:**
```okt
INFERENCE {
    max_tokens: 200
    temperature: 0.7
    top_p: 0.9
    top_k: 40
    repetition_penalty: 1.1
    stop_sequences: ["\n\n", "Human:", "Assistant:"]
}
```

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

```ebnf
<deploy_block> ::= 
  "DEPLOY" "{"
      "target" ":" ("local" | "cloud" | "edge" | "api" | "android" | "ios" | "web" | "desktop")
      [ "endpoint" ":" <string> ]
      [ "requires_auth" ":" ("true" | "false") ]
      [ "port" ":" <number> ]
      [ "max_concurrent_requests" ":" <number> ]
  "}"
```

**Target-specific requirements:**
- `api`: Requires `endpoint` and `port`
- `android`, `ios`: Requires `.okm` or `.tflite` format in EXPORT
- `web`: Requires ONNX format
- `edge`: Requires quantized model (int8 or int4)

**Example:**
```okt
DEPLOY {
    target: "api"
    endpoint: "http://localhost:9000/pizzabot"
    requires_auth: true
    port: 9000
    max_concurrent_requests: 100
}
```

---

## SECURITY Block

```ebnf
<security_block> ::= 
  "SECURITY" "{"
      [ "encrypt_model" ":" ("true" | "false") ]
      [ "watermark" ":" ("true" | "false") ]
      [ "access_token" ":" <string> ]
  "}"
```

**Example:**
```okt
SECURITY {
    encrypt_model: true
    watermark: true
    access_token: "your-secret-token"
}
```

---

## MONITOR Block (v1.1+)

Advanced system and training telemetry monitoring. Extends `METRICS` and `LOGGING` with system-level monitoring.

```ebnf
<monitor_block> ::=
  "MONITOR" "{"
      [<monitor_level>]
      [<monitor_log_metrics>]
      [<monitor_log_system>]
      [<monitor_log_speed>]
      [<monitor_refresh_interval>]
      [<monitor_export_to>]
      [<monitor_dashboard>]
  "}"

<monitor_level> ::=
  "level" ":" ("basic" | "full")

<monitor_log_metrics> ::=
  "log_metrics" ":" "[" <metric_list> "]"

<monitor_log_system> ::=
  "log_system" ":" "[" <system_metric_list> "]"

<monitor_log_speed> ::=
  "log_speed" ":" "[" <speed_metric_list> "]"

<monitor_refresh_interval> ::=
  "refresh_interval" ":" <time_interval>

<monitor_export_to> ::=
  "export_to" ":" <path>

<monitor_dashboard> ::=
  "dashboard" ":" ("true" | "false")

<metric_list> ::=
  <string> { "," <string> }

<system_metric_list> ::=
  ("gpu_memory_used" | "gpu_memory_free" | "cpu_usage" | "ram_used" | "disk_io" | "temperature") { "," <system_metric> }

<speed_metric_list> ::=
  ("tokens_per_second" | "samples_per_second") { "," <speed_metric> }

<time_interval> ::=
  <number> ("s" | "ms")
```

**System Metrics:**
- `gpu_memory_used` / `gpu_memory_free`: GPU memory usage (requires CUDA)
- `cpu_usage`: CPU utilization percentage
- `ram_used`: RAM usage in MB
- `disk_io`: Disk I/O operations per second
- `temperature`: GPU/CPU temperature (if available)

**Speed Metrics:**
- `tokens_per_second`: Token processing speed
- `samples_per_second`: Sample processing speed

**Constraints:**
- `level: "basic"` logs only essential metrics
- `level: "full"` logs all available metrics
- GPU metrics only logged if CUDA is available
- `refresh_interval` must be >= 1s

**Example:**
```okt
MONITOR {
    level: "full"
    log_metrics: [
        "loss",
        "val_loss",
        "accuracy",
        "f1",
        "perplexity"
    ]
    log_system: [
        "gpu_memory_used",
        "gpu_memory_free",
        "cpu_usage",
        "ram_used",
        "temperature"
    ]
    log_speed: [
        "tokens_per_second",
        "samples_per_second"
    ]
    refresh_interval: 2s
    export_to: "runs/logs/system.json"
    dashboard: true
}
```

**Integration with METRICS and LOGGING:**
- `MONITOR` extends (does not replace) `METRICS` and `LOGGING`
- System metrics are logged separately from training metrics
- Dashboard provides real-time visualization (if `dashboard: true`)

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

digit ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
```

**Type constraints:**
- `string`: UTF-8 encoded, max 10,000 characters
- `path`: Can be absolute or relative, must be valid filesystem path
- `number`: Integer, range depends on field (typically 0 to 2^31-1)
- `decimal`: Floating point, precision up to 6 decimal places

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

**Version:** 1.1  
**Last Updated:** November 2025  
**Maintained by:** OktoSeek AI

---

## Version History

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

For more information, visit:
- **Official website:** https://www.oktoseek.com
- **GitHub:** https://github.com/oktoseek/oktoscript
- **Hugging Face:** https://huggingface.co/OktoSeek
- **Twitter:** https://x.com/oktoseek
- **YouTube:** https://www.youtube.com/@Oktoseek
