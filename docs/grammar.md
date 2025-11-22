# OktoScript Grammar Specification v1.0

Complete formal grammar for the OktoScript language.

---

## Grammar Overview

```ebnf
<oktoscript> ::=
  <project_block>
  [<description_block>]
  [<version_block>]
  [<tags_block>]
  [<author_block>]
  <dataset_block>
  <model_block>
  <train_block>
  [<metrics_block>]
  [<validation_block>]
  [<inference_block>]
  [<export_block>]
  [<deploy_block>]
  [<security_block>]
  [<logging_block>]
```

---

## Basic Metadata Blocks

### PROJECT Block

```ebnf
<project_block> ::= 
  "PROJECT" <string>
```

**Example:**
```okt
PROJECT "PizzaBot"
```

### DESCRIPTION Block

```ebnf
<description_block> ::= 
  "DESCRIPTION" <string>
```

**Example:**
```okt
DESCRIPTION "AI specialized in pizza restaurant service"
```

### VERSION Block

```ebnf
<version_block> ::= 
  "VERSION" <string>
```

**Example:**
```okt
VERSION "1.0"
```

### TAGS Block

```ebnf
<tags_block> ::= 
  "TAGS" "[" <string_list> "]"
```

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
      <dataset_train>
      [<dataset_validation>]
      [<dataset_test>]
      [<dataset_format>]
      [<dataset_type>]
      [<dataset_language>]
      [<dataset_augmentation>]
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
```

**Example:**
```okt
DATASET {
  train: "dataset/train.jsonl"
  validation: "dataset/val.jsonl"
  test: "dataset/test.jsonl"
  format: "jsonl"
  type: "chat"
  language: "en"
}
```

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
  "}"

<model_base> ::=
  "base" ":" <string>

<model_architecture> ::=
  "architecture" ":" ("transformer" | "cnn" | "rnn" | "diffusion" | "vision-transformer")

<model_parameters> ::=
  "parameters" ":" <number> ("M" | "B")

<model_context_window> ::=
  "context_window" ":" <number>

<model_precision> ::=
  "precision" ":" ("fp32" | "fp16" | "int8" | "int4")
```

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
  "}"

<train_epochs> ::= 
  "epochs" ":" <number>

<train_batch> ::= 
  "batch_size" ":" <number>

<train_lr> ::= 
  "learning_rate" ":" <decimal>

<train_optimizer> ::= 
  "optimizer" ":" ( "adam" | "adamw" | "sgd" | "rmsprop" )

<train_scheduler> ::= 
  "scheduler" ":" ("linear" | "cosine" | "step")

<train_device> ::= 
  "device" ":" ("cpu" | "cuda" | "mps")

<gradient_accumulation> ::= 
  "gradient_accumulation" ":" <number>

<early_stopping> ::= 
  "early_stopping" ":" ("true" | "false")

<checkpoint_steps> ::= 
  "checkpoint_steps" ":" <number>
```

**Example:**
```okt
TRAIN {
  epochs: 5
  batch_size: 32
  learning_rate: 0.0001
  optimizer: "adamw"
  scheduler: "cosine"
  device: "cuda"
  gradient_accumulation: 2
  checkpoint_steps: 100
  early_stopping: true
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
  "bleu" |
  "rouge" |
  "mae" |
  "mse" |
  "cosine_similarity" |
  "token_efficiency" |
  "response_coherence" |
  "hallucination_score"

<custom_metric> ::= 
  "custom" <string>
```

**Example:**
```okt
METRICS {
  accuracy
  perplexity
  f1
  rouge
  cosine_similarity
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
  "}"
```

**Example:**
```okt
VALIDATE {
  on_train: true
  on_validation: true
  frequency: 1
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
  "}"
```

**Example:**
```okt
INFERENCE {
  max_tokens: 200
  temperature: 0.7
  top_p: 0.9
  top_k: 40
}
```

---

## EXPORT Block

```ebnf
<export_block> ::= 
  "EXPORT" "{"
      "format" ":" "[" <export_format_list> "]"
      "path" ":" <path>
      [ "quantization" ":" ("int8" | "int4" | "fp16") ]
  "}"

<export_format_list> ::= 
  "gguf" |
  "onnx" |
  "okm" |
  "safetensors" |
  "tflite"
```

**Example:**
```okt
EXPORT {
  format: ["gguf", "onnx", "okm", "safetensors"]
  path: "export/"
  quantization: "int8"
}
```

---

## DEPLOY Block

```ebnf
<deploy_block> ::= 
  "DEPLOY" "{"
      "target" ":" ("local" | "cloud" | "edge" | "api")
      [ "endpoint" ":" <string> ]
      [ "requires_auth" ":" ("true" | "false") ]
  "}"
```

**Example:**
```okt
DEPLOY {
  target: "api"
  endpoint: "http://localhost:9000/pizzabot"
  requires_auth: true
}
```

---

## SECURITY Block

```ebnf
<security_block> ::= 
  "SECURITY" "{"
      [ "encrypt_model" ":" ("true" | "false") ]
      [ "watermark" ":" ("true" | "false") ]
  "}"
```

**Example:**
```okt
SECURITY {
  encrypt_model: true
  watermark: true
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
  "}"
```

**Example:**
```okt
LOGGING {
  save_logs: true
  metrics_file: "runs/pizzabot-v1/metrics.json"
  training_file: "runs/pizzabot-v1/training_logs.json"
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

---

## Complete Example

See [`../examples/pizzabot/scripts/train.okt`](../examples/pizzabot/scripts/train.okt) for a complete working example.

---

**Version:** 1.0  
**Last Updated:** 2024  
**Maintained by:** OktoSeek

