<p align="center">
  <img src="./assets/okto_logo.png" alt="OktoScript Banner" width="50%" />
</p>
<p align="center">
  <img src="./assets/okto_logo2.png" alt="OktoScript Banner" width="50%" />
</p>



<h1 align="center">OktoScript</h1>



<p align="center">
  <strong>Domain-specific language for AI training, evaluation and deployment</strong>
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
11. [VS Code Extension](#-vs-code-extension-coming-soon)
12. [Documentation](#-documentation)
13. [FAQ](#-frequently-asked-questions-faq)
14. [License](#-license)
15. [Contact](#-contact)

---

## 🚀 Quick Start

**New to OktoScript?** Get started in 5 minutes:

1. **Read the guide:** [`docs/GETTING_STARTED.md`](./docs/GETTING_STARTED.md)
2. **Try an example:** [`examples/basic.okt`](./examples/basic.okt)
3. **Validate:** `okto validate examples/basic.okt`
4. **Train:** `okto run examples/basic.okt`

📚 **Full documentation:** [`docs/grammar.md`](./docs/grammar.md)  
🔍 **Validation rules:** [`VALIDATION_RULES.md`](./VALIDATION_RULES.md)

---

## 🚀 What is OktoScript?

**OktoScript** is a domain-specific programming language created by **OktoSeek AI** to build, train, evaluate and export AI models in a **structured, readable and repeatable way**.

Designed to be:

- ✅ **Human-readable** - Clear syntax that anyone can understand
- ✅ **Strongly structured** - Type-safe and validated configurations
- ✅ **Dataset-centered** - Built around your data from day one
- ✅ **Training-oriented** - Optimized for ML workflows
- ✅ **Compatible** - Works with modern AI frameworks
- ✅ **Expandable** - Extensible through the OktoEngine

OktoScript is the official language of the OktoSeek ecosystem and is used by:

- 🎯 **OktoSeek IDE** - Visual development environment
- ⚙️ **OktoEngine** - Core execution engine
- 🔌 **VS Code Extension** - Editor integration
- 🔄 **Model pipelines** - Automated workflows
- 📱 **Flutter / API plugins** - Cross-platform deployment

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
  device: "cuda"
}

EXPORT {
  format: ["gguf"]
  path: "export/"
}
```

**Example (v1.1 - LoRA Fine-tuning with Dataset Mixing):**
```okt
# okto_version: "1.1"
PROJECT "PizzaBot"
DESCRIPTION "AI specialized in pizza restaurant service"

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
  device: "cuda"
}

MONITOR {
  level: "full"
  log_metrics: ["loss", "accuracy"]
  log_system: ["gpu_memory_used", "cpu_usage"]
  refresh_interval: 2s
  dashboard: true
}

EXPORT {
  format: ["okm"]
  path: "export/"
}
```

📘 **Full grammar specification available in** [`/docs/grammar.md`](./docs/grammar.md)

## 🆕 What's New in v1.1

OktoScript v1.1 adds powerful new features while maintaining 100% backward compatibility with v1.0:

- ✅ **LoRA Fine-tuning** - Efficient fine-tuning with `FT_LORA` block
- ✅ **Dataset Mixing** - Combine multiple datasets with weighted sampling
- ✅ **System Monitoring** - Advanced telemetry with `MONITOR` block
- ✅ **Version Declaration** - Specify OktoScript version in your files

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
- ✅ **Multi-modal** - (future support)

### Example (JSONL):

```json
{"input":"What flavors do you have?","output":"We offer Margherita, Pepperoni and Four Cheese."}
{"input":"Do you deliver?","output":"Yes, delivery is available in your region."}
```

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

The OktoEngine provides a complete CLI interface for working with OktoScript files. These commands are available both in the terminal and are called by the OktoSeek IDE.

### Main Commands

**Run complete pipeline:**
```bash
# Executes the entire .okt file: dataset → model → train → evaluate → infer → deploy
okto run pizzabot.okt
```

**Train a model:**
```bash
okto_train --config pizzabot.okt
```

**Run inference:**
```bash
okto_infer --model ./models/pizzabot-v1 --text "Boa noite, quero uma pizza grande"

# Or chat mode:
okto_infer --model pizzabot-v1 --chat
```

**Evaluate a model:**
```bash
okto_eval --model ./models/pizzabot-v1 --dataset ./datasets/test.jsonl
```

**Convert formats:**
```bash
okto_convert --from pt --to gguf --input ./models/pizzabot-v1.pt --output ./models/pizzabot-v1.gguf
```

**Validate syntax:**
```bash
okto_validate pizzabot.okt
```

**Deploy model:**
```bash
okto_deploy --model pizzabot-v1 --target api --port 8080
okto_deploy --model pizzabot-v1 --target android
```

**List resources:**
```bash
okto_list projects
okto_list models
okto_list datasets
```

**System diagnostics:**
```bash
okto_doctor
# Shows: GPU, CUDA, RAM, Drivers, Disks, Recommendations
```

### Quick Examples:

```bash
# Validate and train
okto validate examples/basic.okt
okto train examples/chatbot.okt

# Evaluate and export
okto eval examples/recommender.okt
okto export examples/computer_vision.okt --format=okm
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

## 📦 VS Code Extension (Coming Soon)

- ✨ **Syntax Highlighting** - Beautiful code colors
- 🔍 **OktoScript autocomplete** - Smart suggestions
- ⚠️ **Error checking** - Real-time validation
- ▶️ **Run / Train buttons** - One-click execution
- 🎨 **Visual pipeline builder** - Drag-and-drop workflows

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

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

## 📧 Contact

If you have any questions, please raise an issue or contact us at **service@oktoseek.com**.

---

<p align="center">
  Made with ❤️ by the <strong>OktoSeek AI</strong> team
</p>
