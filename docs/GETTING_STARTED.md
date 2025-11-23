# Getting Started with OktoScript

**Your first 5 minutes with OktoScript** - A quick guide to get you up and running.

---

## Prerequisites

- OktoSeek IDE installed (or OktoEngine CLI)
- Basic understanding of AI/ML concepts
- A dataset ready for training

---

## Step 1: Create Your First Project

Create a new directory for your project:

```bash
mkdir my-first-model
cd my-first-model
```

Create a file named `train.okt`:

```okt
PROJECT "MyFirstModel"
DESCRIPTION "My first OktoScript project"

DATASET {
    train: "dataset/train.jsonl"
    format: "jsonl"
    type: "chat"
}

MODEL {
    base: "oktoseek/base-mini"
}

TRAIN {
    epochs: 3
    batch_size: 16
    device: "cpu"
}

EXPORT {
    format: ["okm"]
    path: "export/"
}
```

---

## Step 2: Prepare Your Dataset

Create a `dataset/` folder and add your training data:

**dataset/train.jsonl:**
```json
{"input":"Hello","output":"Hi! How can I help you?"}
{"input":"What's the weather?","output":"I don't have access to weather data."}
{"input":"Thank you","output":"You're welcome!"}
```

**Minimum requirements:**
- At least 10 examples for basic training
- Consistent format (JSONL recommended)
- Valid JSON on each line

---

## Step 3: Validate Your Configuration

Before training, validate your OktoScript file:

```bash
okto validate train.okt
```

This checks:
- ✅ Syntax is correct
- ✅ All required fields are present
- ✅ Dataset files exist
- ✅ Model paths are valid
- ✅ Values are within allowed ranges

---

## Step 4: Train Your Model

Run the training:

```bash
okto run train.okt
```

Or use the IDE:
1. Open `train.okt` in OktoSeek IDE
2. Click "Train" button
3. Monitor progress in real-time

**What happens:**
1. Dataset is loaded and validated
2. Model is initialized
3. Training starts (you'll see progress)
4. Model is saved to `runs/MyFirstModel/`
5. Exported models saved to `export/`

---

## Step 5: Test Your Model

After training, test with inference:

```bash
okto_infer --model ./runs/MyFirstModel --text "Hello"
```

Or add to your `.okt` file:

```okt
INFER {
    input: "Hello, how are you?"
    max_tokens: 50
}
```

---

## Common First Steps

### Adding Validation Data

```okt
DATASET {
    train: "dataset/train.jsonl"
    validation: "dataset/val.jsonl"  # Add this
    format: "jsonl"
}
```

### Using GPU

```okt
TRAIN {
    epochs: 5
    batch_size: 32
    device: "cuda"  # Change from "cpu"
    gpu: true
}
```

### Adding Metrics

```okt
METRICS {
    accuracy
    loss
    perplexity
}
```

### Exporting to Multiple Formats

```okt
EXPORT {
    format: ["gguf", "onnx", "okm"]
    path: "export/"
}
```

---

## Next Steps

- 📚 Read the [Complete Grammar Specification](./grammar.md)
- 🎯 Check out [Complex Examples](../examples/)
- 🔧 Learn about [Troubleshooting](./grammar.md#troubleshooting)
- 💡 Explore [Extension Points](./grammar.md#extension-points--hooks)

---

## Quick Reference

| Task | Command |
|------|---------|
| Validate | `okto validate train.okt` |
| Train | `okto run train.okt` |
| Infer | `okto_infer --model ./runs/model --text "input"` |
| Evaluate | `okto_eval --model ./runs/model --dataset ./dataset/test.jsonl` |
| Export | `okto export --format gguf` |
| Deploy | `okto_deploy --model model --target api` |

---

**Need help?** Check the [Troubleshooting Guide](./grammar.md#troubleshooting) or open an issue on [GitHub](https://github.com/oktoseek/oktoscript).

