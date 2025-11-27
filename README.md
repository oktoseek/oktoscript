<p align="center">
  <img src="./assets/okto_logo.png" alt="OktoScript Banner" width="50%" />
</p>
<p align="center">
  <img src="./assets/okto_logo2.png" alt="OktoScript Banner" width="50%" />
</p>



<h1 align="center">OktoScript</h1>



<p align="center">
  <strong>A decision-driven language for training, evaluating and governing AI models.</strong>
</p>

<p align="center">
  A domain-specific language (DSL) designed for autonomous AI pipelines with<br>
  built-in decision, control, monitoring and governance capabilities.
</p>

<p align="center">
  Built by <strong>OktoSeek AI</strong> for the <strong>OktoSeek ecosystem</strong>
</p>

<p align="center">
  <a href="https://www.oktoseek.com/">OktoSeek Homepage</a> •
  <a href="https://huggingface.co/OktoSeek">Hugging Face</a> •
  <a href="https://x.com/oktoseek">Twitter</a> •
  <a href="https://www.youtube.com/@Oktoseek">YouTube</a>
</p>

---

## Table of Contents

1. [What is OktoScript?](#-what-is-oktoscript)
2. [Quick Start](#-quick-start)
3. [Official Folder Structure](#-official-folder-structure)
4. [Basic Example](#-oktoscript--basic-example)
5. [Supported Dataset Formats](#-supported-dataset-formats)
6. [Supported Metrics](#-supported-metrics)
7. [CLI Commands](#️-cli-commands)
8. [Training Pipeline](#-training-pipeline)
9. [OktoSeek Internal Formats](#-oktoseek-internal-formats)
10. [Integration Targets](#️-integration-targets)
11. [VS Code Extension](#-vs-code-extension)
12. [Documentation](#-documentation)
13. [FAQ](#-frequently-asked-questions-faq)
14. [License](#-license)
15. [Contact](#-contact)

---

## 🚀 Quick Start

**New to OktoScript?** Get started in 5 minutes:

1. **Install VS Code Extension:** [Install OktoScript Extension](https://marketplace.visualstudio.com/items?itemName=OktoSeekAI.oktoscript) (recommended for best experience)
2. **Read the guide:** [`docs/GETTING_STARTED.md`](./docs/GETTING_STARTED.md)
3. **Try an example:** [`examples/basic.okt`](./examples/basic.okt)
4. **Validate:** `okto validate examples/basic.okt`
5. **Train:** `okto train examples/basic.okt`

📚 **Full documentation:** [`docs/grammar.md`](./docs/grammar.md)  
🔍 **Validation rules:** [`VALIDATION_RULES.md`](./VALIDATION_RULES.md)

---

## 🚀 What is OktoScript?

**OktoScript** is a decision-driven language created by **OktoSeek AI** to design, train, evaluate, control and govern AI models end-to-end.

It goes far beyond a simple training script. OktoScript introduces native intelligence, autonomous decision-making and behavioral control into the AI development lifecycle.

It allows you to define:

- **How a model is trained**
- **How it should behave**
- **How it should react to problems**
- **How and when it should stop, adapt or improve itself**

All using clear, readable and structured commands, built specifically for AI engineering.

### Designed to be:

- ✅ **Human-readable** – Intuitive syntax that engineers and non-engineers can understand
- ✅ **Decision-driven** – Built-in CONTROL logic (IF, WHEN, SET, STOP, LOG, SAVE…)
- ✅ **Strongly structured** – Validated, deterministic and reproducible pipelines
- ✅ **Dataset-centered** – The data is the starting point of all intelligence
- ✅ **Training-aware** – Created specifically for AI training and optimization
- ✅ **Behavior-aware** – Control personality, language, restrictions and style
- ✅ **Self-monitoring** – Tracks metrics, detects anomalies and adapts automatically
- ✅ **Safe by design** – Integrated GUARD and SECURITY layers
- ✅ **Expandable** – Extensible through OktoEngine and custom modules

OktoScript is the official language of the OktoSeek ecosystem and is used by:

- 🎯 **OktoSeek IDE** – Visual AI development and experimentation
- ⚙️ **OktoEngine** – Core execution and decision engine
- 🌐 **OktoScript Web Editor** – Online editor with syntax validation and autocomplete ([Try it now →](https://oktoseek.com/editor.php))
- 🔌 **VS Code Extension** – Official VS Code extension with syntax highlighting, autocomplete, snippets, and validation ([Install now →](https://marketplace.visualstudio.com/items?itemName=OktoSeekAI.oktoscript))
- 🔄 **Autonomous pipelines** – Training, control, evaluation and inference
- 🤖 **AI agents** – Controlled, monitored intelligent systems
- 📱 **Flutter / API deployments** – Cross-platform model integration

### Why OktoScript is different

**Traditional AI development is reactive.**  
You manually monitor metrics, fix problems and restart training.

**OktoScript is proactive.**

It allows the model to:

- **Detect instability**
- **Reduce or increase learning rate automatically**
- **Adapt batch size based on GPU memory**
- **Stop when performance drops**
- **Save only the best checkpoints**
- **Apply rules when patterns are detected**

In other words, **OktoScript doesn't just train models — it governs intelligence.**

---

## 📁 Official Folder Structure

Every OktoScript project must follow this structure:

```
/my-awesome-model
├── okt.yaml
├── dataset/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── scripts/
│   └── train.okt
├── runs/
│   └── my-model/
│       ├── checkpoint-100/
│       │   └── model.safetensors
│       ├── tokenizer.json
│       ├── training_logs.json
│       └── metrics.json
└── export/
    ├── model.gguf
    ├── model.onnx
    └── model.okm
```

**v1.1 Optional Folders:**

```
/runs/
  └── my-model/
      ├── logs/
      │   └── system.json      # MONITOR output (v1.1+)
      └── lora/                 # LoRA adapters (v1.1+)
          └── adapter.safetensors
```

---

## 🧠 OktoScript – Basic Example

**Example (v1.0 - Standard Training):**
```okt
PROJECT "PizzaBot"
DESCRIPTION "AI specialized in pizza restaurant service"

ENV {
  accelerator: "gpu"
  min_memory: "8GB"
  precision: "fp16"
  backend: "oktoseek"
  install_missing: true
}

DATASET {
  train: "dataset/train.jsonl"
  validation: "dataset/val.jsonl"
}

MODEL {
  base: "oktoseek/pizza-small"
}

TRAIN {
  epochs: 5
  batch_size: 32
  device: "auto"
}

EXPORT {
  format: ["gguf", "onnx", "okm"]
  path: "export/"
}
```

**Example (v1.1 - LoRA Fine-tuning with Dataset Mixing):**
```okt
# okto_version: "1.1"
PROJECT "PizzaBot"
DESCRIPTION "AI specialized in pizza restaurant service"

ENV {
  accelerator: "gpu"
  min_memory: "8GB"
  precision: "fp16"
  backend: "oktoseek"
  install_missing: true
}

DATASET {
  mix_datasets: [
    { path: "dataset/base.jsonl", weight: 70 },
    { path: "dataset/extra.jsonl", weight: 30 }
  ]
  dataset_percent: 80
  sampling: "weighted"
}

MODEL {
  base: "oktoseek/pizza-small"
}

FT_LORA {
  base_model: "oktoseek/pizza-small"
  lora_rank: 8
  lora_alpha: 32
  epochs: 3
  batch_size: 16
  learning_rate: 0.00003
  device: "auto"
}

MONITOR {
  level: "full"
  log_metrics: ["loss", "accuracy"]
  log_system: ["gpu_memory_used", "cpu_usage"]
  refresh_interval: 2s
  dashboard: true
}

EXPORT {
  format: ["okm", "onnx"]
  path: "export/"
}
```

📘 **Full grammar specification available in** [`/docs/grammar.md`](./docs/grammar.md)

## 🆕 What's New in v1.2

OktoScript v1.2 adds powerful new features while maintaining 100% backward compatibility with v1.0 and v1.1:

- ✅ **Nested CONTROL Blocks** - Support for nested IF/WHEN/EVERY statements inside event hooks
- ✅ **Enhanced BEHAVIOR** - Added `mode` and `prompt_style` for better control
- ✅ **Enhanced GUARD** - Added `detect_using` and additional prevention types
- ✅ **Enhanced DEPLOY** - Added `host`, `protocol`, and `format` options
- ✅ **Enhanced SECURITY** - Added input/output validation, rate limiting, and encryption

## What's New in v1.1

OktoScript v1.1 adds powerful new features while maintaining 100% backward compatibility with v1.0:

- ✅ **LoRA Fine-tuning** - Efficient fine-tuning with `FT_LORA` block
- ✅ **Dataset Mixing** - Combine multiple datasets with weighted sampling
- ✅ **System Monitoring** - Advanced telemetry with `MONITOR` block
- ✅ **Version Declaration** - Specify OktoScript version in your files
- ✅ **MODEL Adapters** - LoRA/PEFT adapter support in MODEL block
- ✅ **Enhanced INFERENCE** - Rich inference configuration with format templates and nested CONTROL
- ✅ **CONTROL Block** - Cognitive-level decision engine for training and inference
- ✅ **GUARD Block** - Safety and ethics protection
- ✅ **BEHAVIOR Block** - Model personality and behavior configuration
- ✅ **EXPLORER Block** - AutoML-style hyperparameter exploration
- ✅ **STABILITY Block** - Training stability and safety controls
- ✅ **Boolean Support** - Native true/false values throughout the language

📚 **More examples and use cases:** See [`/examples/`](./examples/) for complete examples including:

**Basic Examples:**
- [`basic.okt`](./examples/basic.okt) - Minimal example
- [`chatbot.okt`](./examples/chatbot.okt) - Conversational AI
- [`computer_vision.okt`](./examples/computer_vision.okt) - Image classification
- [`recommender.okt`](./examples/recommender.okt) - Recommendation systems

**Advanced Examples:**
- [`finetuning-llm.okt`](./examples/finetuning-llm.okt) - Fine-tuning LLM with checkpoints and hooks
- [`vision-pipeline.okt`](./examples/vision-pipeline.okt) - Complete vision pipeline with augmentation
- [`qa-embeddings.okt`](./examples/qa-embeddings.okt) - QA system with embeddings

**v1.1 Examples:**
- [`lora-finetuning.okt`](./examples/lora-finetuning.okt) - LoRA fine-tuning with dataset mixing
- [`dataset-mixing.okt`](./examples/dataset-mixing.okt) - Training with multiple weighted datasets

**Complete Projects:**
- [`pizzabot/`](./examples/pizzabot/) - Complete project example with full structure

---

## 📚 Supported Dataset Formats

- ✅ **JSONL** - Line-delimited JSON
- ✅ **CSV** - Comma-separated values
- ✅ **TXT** - Plain text files
- ✅ **Parquet** - Columnar storage
- ✅ **Image + Caption** - Vision datasets
- ✅ **Question & Answer (QA)** - Q&A pairs
- ✅ **Instruction datasets** - Instruction-following
- ✅ **Custom Field Names** (v1.2+) - Define `input_field` and `output_field` for any column names
- ✅ **Multi-modal** - (future support)

### Example (JSONL):

```json
{"input":"What flavors do you have?","output":"We offer Margherita, Pepperoni and Four Cheese."}
{"input":"Do you deliver?","output":"Yes, delivery is available in your region."}
```

### Custom Field Names (v1.2+)

OktoScript now supports custom field names in datasets, allowing you to work with any column names:

```okt
DATASET {
    train: "dataset/train.jsonl"
    input_field: "question"    # Custom input column name
    output_field: "answer"      # Custom output column name
}
```

If not specified, OktoEngine automatically detects `input`/`output` or `input`/`target` fields.

📖 **[Learn more about custom fields →](./docs/CUSTOM_FIELDS.md)**

---

## 📊 Supported Metrics

- ✅ **Accuracy** - Classification accuracy
- ✅ **Loss** - Training/validation loss
- ✅ **Perplexity** - Language model perplexity
- ✅ **F1-Score** - F1 metric
- ✅ **BLEU** - Translation quality
- ✅ **ROUGE-L** - Summarization quality
- ✅ **MAE / MSE** - Regression metrics
- ✅ **Cosine Similarity** - Embedding similarity
- ✅ **Token Efficiency** - Token usage optimization
- ✅ **Response Coherence** - Response quality
- ✅ **Hallucination Score** - (experimental)

### Define custom metrics:

```okt
METRICS {
  custom "toxicity_score"
  custom "context_alignment"
}
```

---

## 🖥️ CLI Commands

The OktoEngine CLI is minimal by design. All intelligence lives in the `.okt` file. The terminal is just the execution port.

### 🌐 Web Editor Command

**Open OktoScript files in the web editor:**

```bash
# Open editor with a specific file
okto web --file scripts/train.okt

# Open empty editor
okto web
```

The `okto web` command opens the [OktoScript Web Editor](https://oktoseek.com/editor.php) in your browser. When you provide a file path, it automatically loads the file content for editing. The editor features:

- **Smart Autocomplete** – Context-aware suggestions based on the current block (ENV, DATASET, MODEL, TRAIN, etc.)
- **Real-time Syntax Validation** – Detects errors like nested blocks (e.g., PROJECT inside DATASET) and missing braces
- **Auto-save to Local** – When you load a file, it saves back to the same location automatically
- **Full Integration** – Seamlessly connects with OktoEngine for validation and training

Perfect for quick edits, syntax testing, and experimenting with OktoScript configurations!

### Core Commands

**Initialize a project:**
```bash
okto init
```

**Validate syntax:**
```bash
okto validate script.okt
```

**Train a model:**
```bash
okto train script.okt
```

**Evaluate a model:**
```bash
okto eval script.okt
```

**Export model:**
```bash
okto export script.okt
```

**Convert model formats:**
```bash
okto convert --input <model_path> --from <format> --to <format> --output <output_path>
```

**Supported formats:**
| From / To | Usage |
|-----------|-------|
| `pt`, `bin` | PyTorch |
| `onnx` | Web / Interoperability |
| `tflite` | Mobile (Android / iOS) |
| `gguf` | Local LLMs (llama.cpp) |
| `okm` | Okto Model Format |
| `safetensors` | Safe and fast |

**Convert examples:**
```bash
# PyTorch → GGUF (local inference)
okto convert --input model.pt --from pt --to gguf --output model.gguf

# PyTorch → TFLite (mobile)
okto convert --input model.pt --from pt --to tflite --output model.tflite

# PyTorch → ONNX (web)
okto convert --input model.pt --from pt --to onnx --output model.onnx
```

**List resources:**
```bash
okto list projects
okto list models
okto list datasets
okto list exports
```

**System diagnostics:**
```bash
okto doctor
# Shows: GPU, CUDA, RAM, Drivers, Disks, Recommendations
```

### Inference Commands

**Direct inference (single input/output):**
```bash
okto infer --model <model_path> --text "<input>"
```

**Example:**
```bash
okto infer --model models/pizzabot.okm --text "Good evening, I want a pizza"
```

Automatically respects:
- `BEHAVIOR` block
- `GUARD` block
- `INFERENCE` block
- `CONTROL` block (if defined)

**Interactive chat mode:**
```bash
okto chat --model <model_path>
```

Opens an interactive loop:
```
🟢 Okto Chat started (type 'exit' to quit)

You: hi
Bot: Hello! How can I help you?

You: what flavors do you have?
Bot: We have...

You: exit
🔴 Session ended
```

This command:
- Uses `prompt_style` from BEHAVIOR
- Uses `BEHAVIOR` settings
- Respects `GUARD` rules
- Can use MEMORY in the future

### Advanced Commands

**Compare two models:**
```bash
okto compare <model1> <model2>
```

**Example:**
```bash
okto compare models/pizza_v1.okm models/pizza_v2.okm
```

Expected output:
```
Latency: V2 - 23% faster
Accuracy: V1 - 4% better
Loss: V2 - lower
Recommendation: V2
```

Perfect for A/B testing.

**View historical logs:**
```bash
okto logs <model_or_run_id>
```

**Example:**
```bash
okto logs pizzabot_v1
```

Shows:
- Loss per epoch
- Validation loss
- Accuracy
- CPU/GPU/RAM usage
- Decisions made (CONTROL block)

**Auto-tune training:**
```bash
okto tune script.okt
```

Uses the `CONTROL` block to auto-adjust training based on metrics. Can:
- Adjust learning rate
- Change batch size
- Activate early stopping
- Balance classes

This is unique in the market.

**Exit interactive mode:**
```bash
okto exit
```

Used to exit chat, interactive mode, or session context.

### Utility Commands

```bash
okto upgrade    # Update OktoEngine
okto about     # Show about information
okto --version # Show version
```

### Quick Examples

```bash
# Validate and train
okto validate examples/basic.okt
okto train examples/chatbot.okt

# Evaluate and export
okto eval examples/recommender.okt
okto export examples/computer_vision.okt

# Inference
okto infer --model models/bot.okm --text "Hello"
okto chat --model models/bot.okm
```

---

## 🔄 Training Pipeline

1. **Load dataset** - Parse and validate input data
2. **Tokenize & validate** - Prepare data for training
3. **Initialize model** - Load base model and configuration
4. **Train loop** - Execute training epochs
5. **Calculate metrics** - Evaluate model performance
6. **Export selected models** - Generate output formats
7. **Generate final report** - Create training summary

Each run generates logs at:

```
runs/my-model/training_logs.json
runs/my-model/metrics.json
```

---

## 🔐 Export Formats

### Standard Formats

| Format | Purpose | Compatibility |
|--------|---------|---------------|
| `.onnx` | Universal inference, production-ready | All platforms |
| `.gguf` | Local inference, Ollama, Llama.cpp | Local deployment |
| `.safetensors` | HuggingFace, research, training | Standard ML tools |
| `.tflite` | Mobile deployment | Android, iOS (future) |

### OktoSeek Optimized Formats

| Format | Purpose | Benefits |
|--------|---------|----------|
| `.okm` | **OktoModel** - Optimized for OktoSeek SDK | Flutter plugins, mobile apps, exclusive tools |
| `.okx` | **OktoBundle** - Mobile + Edge package | iOS, Android, Edge AI deployment |

> 💡 **Note:** `.okm` and `.okx` formats are **optional** and optimized for the OktoSeek ecosystem. They provide better integration with OktoSeek Flutter SDK, mobile apps, and exclusive tools. You can always export to standard formats (ONNX, GGUF, SafeTensors) for universal compatibility.

**Why use OktoModel (.okm)?**

- ✅ Optimized for OktoSeek Flutter SDK
- ✅ Better performance on mobile devices
- ✅ Access to exclusive OktoSeek tools and plugins
- ✅ Seamless integration with OktoSeek ecosystem
- ✅ Support for iOS and Android apps

See [`/examples/`](./examples/) for examples using different export formats.

---

## ⚙️ Integration Targets

- ✅ **Flutter** - Mobile applications
- ✅ **REST API** - Web services
- ✅ **Edge AI** - Edge devices
- ✅ **Desktop** - Native applications
- ✅ **Web** - Browser-based
- ✅ **Mobile** - iOS/Android
- ✅ **IoT** - Internet of Things
- ✅ **Robotics** - Robotic systems

---

## 📦 VS Code Extension

**Official OktoScript extension for Visual Studio Code is now available!**

[![Install OktoScript Extension](https://img.shields.io/badge/VS%20Code-Install%20OktoScript-blue?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=OktoSeekAI.oktoscript)

### Features

- ✨ **Syntax Highlighting** - Beautiful color-coded OktoScript syntax for all blocks, keywords, and values
- 🔍 **Smart Autocomplete** - Context-aware suggestions based on the current block (ENV, DATASET, MODEL, TRAIN, etc.)
- 📝 **Code Snippets** - Quick templates for all OktoScript blocks (PROJECT, MODEL, TRAIN, CONTROL, INFERENCE, FT_LORA, etc.)
- ✅ **Syntax Validation** - Validate your `.okt` files using OktoEngine directly from VS Code
- 🌐 **Web Editor Integration** - Open files directly in the OktoScript Web Editor with one command
- 🎯 **Intelligent Suggestions** - Autocomplete triggers automatically on typing or pressing space
- 📚 **Block Templates** - Selecting a block from autocomplete generates a complete template (e.g., `MODEL { }`)

### Installation

**From VS Code Marketplace:**
1. Open VS Code
2. Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on Mac) to open Extensions
3. Search for "OktoScript"
4. Click "Install"

**Or use command line:**
```bash
code --install-extension OktoSeekAI.oktoscript
```

**Direct Link:** [Install OktoScript Extension](https://marketplace.visualstudio.com/items?itemName=OktoSeekAI.oktoscript)

### Usage

- **Syntax Highlighting:** Open any `.okt` file and enjoy beautiful syntax highlighting
- **Autocomplete:** Start typing a block name (e.g., `MODEL`, `TRAIN`) and see contextual suggestions
- **Snippets:** Type block names and press `Tab` to insert complete templates
- **Validation:** Press `Ctrl+Shift+P` → "OktoScript: Validate current file" (requires OktoEngine)
- **Web Editor:** Press `Ctrl+Shift+P` → "OktoScript: Open in Web Editor" (requires OktoEngine)

> 💡 **Tip:** The VS Code extension works seamlessly with the [🌐 OktoScript Web Editor](https://oktoseek.com/editor.php). Both provide context-aware autocomplete, real-time syntax validation, and full integration with OktoEngine via the `okto web` command!

---

## 📚 Documentation

Complete documentation for OktoScript:

- 📖 **[Grammar Specification](./docs/grammar.md)** - Complete formal grammar with all constraints (v1.0 & v1.1)
- 🚀 **[Getting Started Guide](./docs/GETTING_STARTED.md)** - Your first 5 minutes with OktoScript
- ✅ **[Validation Rules](./VALIDATION_RULES.md)** - Complete validation reference (updated for v1.1)
- ❓ **[FAQ](./docs/FAQ.md)** - Frequently Asked Questions - Common questions and detailed answers
- 💡 **[Examples](./examples/)** - Working examples from basic to advanced
- 📋 **[Changelog v1.1](./CHANGELOG_V1.1.md)** - Complete list of v1.1 features

### Advanced Topics

- 🔗 **[Model Inheritance](./docs/grammar.md#model-inheritance)** - Reuse model configurations
- 🔌 **[Extension Points & Hooks](./docs/grammar.md#extension-points--hooks)** - Custom Python/JS integration
- 🐛 **[Troubleshooting](./docs/grammar.md#troubleshooting)** - Common issues and solutions
- ⚙️ **[Complex Examples](./examples/)** - Advanced use cases:
  - [`finetuning-llm.okt`](./examples/finetuning-llm.okt) - Fine-tuning with checkpoints
  - [`vision-pipeline.okt`](./examples/vision-pipeline.okt) - Production vision systems
  - [`qa-embeddings.okt`](./examples/qa-embeddings.okt) - Semantic search and retrieval
  - [`lora-finetuning.okt`](./examples/lora-finetuning.okt) - LoRA fine-tuning (v1.1)
  - [`dataset-mixing.okt`](./examples/dataset-mixing.okt) - Dataset mixing (v1.1)

---

## ❓ Frequently Asked Questions (FAQ)

Have questions about OktoScript? Check out our comprehensive FAQ covering common questions from beginners to advanced users:

**Common Questions:**
- Why do I need MODEL and DATASET blocks with FT_LORA?
- What's the difference between FT_LORA and TRAIN?
- Does OktoScript replace Python?
- How do I use multiple datasets with weights?
- Can I use custom Python code?
- Is OktoScript a programming language or a DSL?
- And 15+ more detailed answers...

📖 **[Read the complete FAQ →](./docs/FAQ.md)**

The FAQ covers technical details, design decisions, use cases, and best practices for using OktoScript effectively.

---

## 🧑‍🚀 Vision

> *"Knowledge must be shared between people so that we can create solutions we could never imagine."*
> 
> — **OktoSeek AI**

### 🎯 Design Principles

OktoScript is built on the principle that AI development should be:

1. **Declarative** - Describe what you want, not how to do it
2. **Self-aware** - Models can monitor and adjust themselves
3. **Safe** - Built-in guards against harmful outputs
4. **Adaptive** - Automatic optimization and exploration
5. **Transparent** - Clear, readable configuration files
6. **Powerful** - Complex capabilities with simple syntax

The language evolves to support increasingly sophisticated AI behaviors while maintaining its core simplicity.

---

## 🐙 Powered by OktoSeek AI

**OktoScript** is developed and maintained by **OktoSeek AI**.

- **Official website:** https://www.oktoseek.com
- **GitHub:** https://github.com/oktoseek
- **Hugging Face:** https://huggingface.co/OktoSeek
- **Twitter:** https://x.com/oktoseek
- **YouTube:** https://www.youtube.com/@Oktoseek
- **Repository:** https://github.com/oktoseek/oktoscript

---

## 📄 License

OktoScript is available for personal and commercial use at no cost.

However, OktoScript is a proprietary language owned by OktoSeek AI and may not be modified or used to create derivative languages, tools or interpreters.

See [OKTOSCRIPT_LICENSE.md](./OKTOSCRIPT_LICENSE.md) for complete license terms.

---

## 🤝 Contributing

Contributions are welcome! We welcome bug reports, feature suggestions, documentation improvements, and example contributions. Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

**Note:** OktoScript is a proprietary language owned by OktoSeek AI. While we welcome contributions, you may not create derivative languages, tools, or interpreters based on OktoScript.

---

## 📧 Contact

If you have any questions, please raise an issue or contact us at **service@oktoseek.com**.

---

<p align="center">
  Made with ❤️ by the <strong>OktoSeek AI</strong> team
</p>
