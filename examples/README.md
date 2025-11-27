# OktoScript Examples

This folder contains official example scripts written in **OktoScript (.okt)**.

These examples are used by:

- Developers learning OktoScript
- Students and researchers
- OktoSeek IDE
- VS Code Extension
- Automatic tests and validation

---

## Available Examples

### Basic Examples

| File | Description | Use Case |
|------|-------------|----------|
| [`basic.okt`](./basic.okt) | Minimal example | Getting started |
| [`chatbot.okt`](./chatbot.okt) | Conversational AI | Customer service, assistants |
| [`computer_vision.okt`](./computer_vision.okt) | Image classification | Vision models, object detection |
| [`recommender.okt`](./recommender.okt) | Recommendation system | E-commerce, content suggestions |

### Advanced Examples

| File | Description | Use Case |
|------|-------------|----------|
| [`finetuning-llm.okt`](./finetuning-llm.okt) | Fine-tuning LLM with checkpoints | Advanced language models, resume training |
| [`vision-pipeline.okt`](./vision-pipeline.okt) | Complete vision pipeline | Production vision systems, ONNX export |
| [`qa-embeddings.okt`](./qa-embeddings.okt) | QA with embeddings | Semantic search, retrieval systems |

### v1.1 Examples (New Features)

| File | Description | Use Case |
|------|-------------|----------|
| [`lora-finetuning.okt`](./lora-finetuning.okt) | LoRA fine-tuning with dataset mixing | Efficient fine-tuning, memory-efficient training |
| [`dataset-mixing.okt`](./dataset-mixing.okt) | Training with multiple weighted datasets | Combining datasets, weighted sampling |

### 🧪 Test Scripts (Recommended for Testing)

These scripts are specifically designed for testing different features of OktoScript v1.2:

| File | Description | Features Tested |
|------|-------------|-----------------|
| [`test-t5-basic.okt`](./test-t5-basic.okt) | Basic training | PROJECT, ENV, DATASET, MODEL, TRAIN, EXPORT |
| [`test-t5-monitor.okt`](./test-t5-monitor.okt) | Training with MONITOR | Full metrics tracking, notifications |
| [`test-t5-control.okt`](./test-t5-control.okt) | Training with CONTROL | Automatic decisions, IF/WHEN/EVERY |
| [`test-flan-t5-complete.okt`](./test-flan-t5-complete.okt) | All advanced blocks | MONITOR, CONTROL, STABILITY together |
| [`test-flan-t5-inference.okt`](./test-flan-t5-inference.okt) | Inference with governance | BEHAVIOR, GUARD, INFERENCE blocks |
| [`test-t5-explorer.okt`](./test-t5-explorer.okt) | AutoML with EXPLORER | Hyperparameter search, best model selection |

📖 **See [`TESTING_GUIDE.md`](./TESTING_GUIDE.md) for detailed testing instructions.**

---

### v1.2 Examples (Advanced Features)

| File | Description | Use Case |
|------|-------------|----------|
| [`control-nested.okt`](./control-nested.okt) | Nested CONTROL blocks with advanced decision-making | Dynamic training control, conditional logic |
| [`behavior-chat.okt`](./behavior-chat.okt) | BEHAVIOR block with mode and prompt_style | Chatbot personality, response style |
| [`guard-safety.okt`](./guard-safety.okt) | GUARD block with multiple detection methods | Content safety, ethical AI |
| [`deploy-api.okt`](./deploy-api.okt) | DEPLOY block for API deployment | Production API deployment |
| [`security-full.okt`](./security-full.okt) | Complete SECURITY block configuration | Input/output validation, rate limiting |
| [`model-adapter.okt`](./model-adapter.okt) | MODEL block with ADAPTER (LoRA/PEFT) | Parameter-efficient fine-tuning |
| [`inference-advanced.okt`](./inference-advanced.okt) | Advanced INFERENCE with nested CONTROL | Smart inference with retry logic |
| [`monitor-full.okt`](./monitor-full.okt) | Complete MONITOR block with all metrics | Full system and training telemetry |
| [`explorer-automl.okt`](./explorer-automl.okt) | EXPLORER block for hyperparameter search | AutoML-style optimization |
| [`stability-training.okt`](./stability-training.okt) | STABILITY block for safe training | Training stability and safety |
| [`complete-v1.2.okt`](./complete-v1.2.okt) | Complete example with all v1.2 features | Full feature demonstration |

### Complete Projects

| File | Description | Use Case |
|------|-------------|----------|
| [`pizzabot/`](./pizzabot/) | Complete project example | Full workflow demonstration |

---

## Quick Start

To run these examples with OktoEngine (when available):

```bash
# Validate syntax
okto validate examples/basic.okt

# Train a model
okto train examples/chatbot.okt

# Evaluate performance
okto eval examples/recommender.okt

# Export model
okto export examples/computer_vision.okt --format=okm
```

---

## Export Formats

OktoScript supports multiple export formats for different use cases:

### Standard Formats

- **ONNX** - Universal inference, production-ready
- **GGUF** - Local inference, Ollama, Llama.cpp
- **SafeTensors** - HuggingFace, research, standard training

### OktoSeek Optimized Formats

- **OktoModel (.okm)** - Optimized for OktoSeek SDK & Flutter plugins
- **OktoBundle (.okx)** - Mobile + Edge package (iOS, Android, Edge AI)

> 💡 **Tip:** While standard formats work everywhere, `.okm` and `.okx` formats are optimized for the OktoSeek ecosystem, providing better integration with Flutter apps, mobile SDKs, and OktoSeek tools.

---

## Example: Using OktoModel Format

```okt
EXPORT {
  format: ["onnx", "okm"]
  path: "export/"
}
```

**Why use .okm?**

- ✅ Optimized for OktoSeek Flutter SDK
- ✅ Better performance on mobile devices
- ✅ Access to exclusive OktoSeek tools and plugins
- ✅ Seamless integration with OktoSeek ecosystem
- ✅ Support for iOS and Android apps

**Note:** `.okm` is optional. You can always export to standard formats (ONNX, GGUF, SafeTensors) for universal compatibility.

---

## Training Workflow

During training, OktoScript uses standard formats (this is industry-standard):

```
runs/my-model/
├── checkpoint-100/
│   └── model.safetensors
├── checkpoint-200/
│   └── model.safetensors
├── tokenizer.json
└── training_logs.json
```

After training, you choose your export format based on your deployment needs.

---

## Complete Project Example

See [`pizzabot/`](./pizzabot/) for a complete project example with:
- Full project structure
- Multiple dataset files
- Training configuration
- Export settings
- Example outputs

---

## Contributing

Want to add your own example? 

1. Create a new `.okt` file in this directory
2. Follow the OktoScript grammar specification
3. Include clear comments and descriptions
4. Submit a pull request!

---

**Powered by OktoSeek AI**

- **Website:** https://www.oktoseek.com
- **GitHub:** https://github.com/oktoseek/oktoscript
- **Documentation:** [../docs/grammar.md](../docs/grammar.md)
