# OktoScript Examples

This directory contains example projects demonstrating OktoScript usage. **OktoScript** is a domain-specific programming language developed by **OktoSeek AI**.

## Available Examples

### 🍕 PizzaBot

A complete chatbot example for a pizza restaurant service, developed using **OktoScript** by **OktoSeek AI**.

**Location:** `pizzabot/`

**Files:**
- `scripts/train.okt` - Complete training configuration
- `okt.yaml` - Project configuration
- `dataset/` - Training, validation, and test datasets
- `runs/` - Example training outputs and metrics

**Quick Start:**
```bash
cd pizzabot
okto validate
okto train
```

📝 **Complete training script:** [`pizzabot/scripts/train.okt`](./pizzabot/scripts/train.okt)

📊 **Example datasets:**
- Training: [`pizzabot/dataset/train.jsonl`](./pizzabot/dataset/train.jsonl)
- Validation: [`pizzabot/dataset/val.jsonl`](./pizzabot/dataset/val.jsonl)
- Test: [`pizzabot/dataset/test.jsonl`](./pizzabot/dataset/test.jsonl)

---

## Creating Your Own Example

1. Create a new directory with your project name
2. Follow the standard OktoScript folder structure
3. Add a `train.okt` file in the `scripts/` directory
4. Include sample datasets in the `dataset/` directory
5. Submit a pull request to add your example here!

---

## About OktoScript

**OktoScript** is developed and maintained by **OktoSeek AI**. It is the official language of the OktoSeek ecosystem, used by OktoSeek IDE, OktoEngine, and various AI development tools.

For more information, see the [main README](../README.md) and [grammar documentation](../docs/grammar.md).

---

**Powered by OktoSeek AI**
