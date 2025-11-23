# OktoScript Validation Rules

Complete reference for validation rules and constraints in OktoScript.

---

## File Structure Validation

### Required Files

1. **okt.yaml** (in project root)
   - Must exist
   - Must be valid YAML
   - Must contain `project` field

2. **Dataset Files**
   - All paths specified in DATASET block must exist
   - Files must be readable
   - Format must match declared format

3. **Model Files** (if using local paths)
   - Base model path must exist (if local)
   - Checkpoint paths must exist (if resuming)

---

## Field Validation

### PROJECT Block

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| PROJECT | string | ✅ Yes | 1-100 chars, no special chars: `{}[]:"` |

### DATASET Block

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| train | path | ✅ Yes | File/dir must exist, readable |
| validation | path | ❌ No | File/dir must exist if specified |
| test | path | ❌ No | File/dir must exist if specified |
| format | enum | ❌ No | Must be: jsonl, csv, txt, parquet, image+caption, qa, instruction, multimodal |
| type | enum | ❌ No | Must be: classification, generation, qa, chat, vision, regression |
| language | enum | ❌ No | Must be: en, pt, es, fr, multilingual |
| augmentation | array | ❌ No | Each item must be valid augmentation type |

### MODEL Block

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| base | string | ✅ Yes | Valid model identifier or path |
| architecture | enum | ❌ No | Must be: transformer, cnn, rnn, diffusion, vision-transformer, bert, gpt, t5 |
| parameters | string | ❌ No | Format: number + (K\|M\|B), e.g., "120M" |
| context_window | number | ❌ No | Must be power of 2: 128, 256, 512, 1024, 2048, 4096, 8192 |
| precision | enum | ❌ No | Must be: fp32, fp16, int8, int4 |
| inherit | string | ❌ No | Must reference existing model name |

### TRAIN Block

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| epochs | number | ✅ Yes | > 0 and <= 1000 |
| batch_size | number | ✅ Yes | > 0 and <= 1024 |
| learning_rate | decimal | ❌ No | > 0 and <= 1.0 |
| optimizer | enum | ❌ No | Must be: adam, adamw, sgd, rmsprop, adafactor, lamb |
| scheduler | enum | ❌ No | Must be: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup, step |
| device | enum | ✅ Yes | Must be: cpu, cuda, mps, auto |
| gradient_accumulation | number | ❌ No | >= 1 |
| early_stopping | boolean | ❌ No | true or false |
| checkpoint_steps | number | ❌ No | > 0 |
| checkpoint_path | path | ❌ No | Directory must exist if specified |
| resume_from_checkpoint | path | ❌ No | Checkpoint must exist if specified |
| loss | enum | ❌ No | Must be: cross_entropy, mse, mae, bce, focal, huber, kl_divergence |
| weight_decay | decimal | ❌ No | >= 0 and <= 1.0 |
| gradient_clip | decimal | ❌ No | > 0 |
| warmup_steps | number | ❌ No | >= 0 |
| save_strategy | enum | ❌ No | Must be: steps, epoch, no |

### METRICS Block

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| Built-in metrics | identifier | ❌ No | Must be valid metric name |
| custom | string | ❌ No | Custom metric identifier |

**Metric-task compatibility:**
- `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`: Only for classification
- `perplexity`: Only for language models
- `bleu`, `rouge`: Only for generation/translation
- `mae`, `mse`, `rmse`: Only for regression

### EXPORT Block

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| format | array | ✅ Yes | Each item must be: gguf, onnx, okm, safetensors, tflite |
| path | path | ✅ Yes | Directory must exist or be creatable |
| quantization | enum | ❌ No | Must be: int8, int4, fp16, fp32 |
| optimize_for | enum | ❌ No | Must be: speed, size, accuracy |

**Format-specific requirements:**
- `gguf`: Requires quantization
- `tflite`: Only for mobile-compatible architectures

### DEPLOY Block

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| target | enum | ✅ Yes | Must be: local, cloud, edge, api, android, ios, web, desktop |
| endpoint | string | ❌ No | Required if target is "api" |
| requires_auth | boolean | ❌ No | true or false |
| port | number | ❌ No | Required if target is "api", must be 1024-65535 |
| max_concurrent_requests | number | ❌ No | > 0 |

---

## Dependency Validation

### Model Inheritance

- If `inherit` is specified, parent model must be defined
- Circular inheritance is not allowed
- Inheritance chain depth limited to 10 levels

### Checkpoint Resume

- If `resume_from_checkpoint` is specified:
  - Checkpoint directory must exist
  - Checkpoint must contain valid model files
  - Checkpoint must be compatible with current model architecture

### Export Compatibility

- Model architecture must support export format
- Quantization required for certain formats (gguf)
- Mobile formats (tflite, okm) require compatible architectures

---

## Runtime Validation

### Dataset Validation

**File existence:**
- All dataset paths must exist
- Files must be readable
- Directories must be accessible

**Format validation:**
- JSONL: Each line must be valid JSON
- CSV: Must have header row, consistent columns
- Image+caption: Directory must contain image files and captions

**Size limits:**
- Maximum file size: 10GB per file
- Maximum total dataset size: 100GB
- Minimum examples: 10 for training

### Model Validation

**Base model:**
- If local path: Must exist and be valid model directory
- If HuggingFace: Must be downloadable
- If URL: Must be accessible

**Architecture compatibility:**
- Model architecture must match dataset type
- Vision models require image datasets
- Language models require text datasets

### Training Validation

**Hardware requirements:**
- GPU required if `device: "cuda"` and `gpu: true`
- Sufficient VRAM for batch size
- Sufficient disk space for checkpoints

**Memory validation:**
- Batch size must fit in available memory
- Effective batch size (batch_size × gradient_accumulation) validated

---

## Error Codes

| Code | Error | Solution |
|------|-------|----------|
| V001 | Dataset file not found | Check file path, use absolute or relative path |
| V002 | Invalid optimizer | Use one of: adam, adamw, sgd, rmsprop, adafactor, lamb |
| V003 | Invalid scheduler | Use one of: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup, step |
| V004 | Model base not found | Verify model path or HuggingFace model name |
| V005 | Checkpoint not found | Check checkpoint path or remove resume_from_checkpoint |
| V006 | Insufficient memory | Reduce batch_size or enable gradient_accumulation |
| V007 | Invalid metric for task | Use appropriate metrics for task type |
| V008 | Invalid export format | Check format compatibility with model architecture |
| V009 | Circular inheritance | Remove circular model inheritance chain |
| V010 | Invalid field value | Check field constraints and allowed values |

---

## Validation Commands

### CLI Validation

```bash
# Validate syntax and structure
okto validate train.okt

# Validate with detailed output
okto validate train.okt --verbose

# Validate dataset only
okto validate train.okt --dataset-only

# Validate model only
okto validate train.okt --model-only
```

### IDE Validation

OktoSeek IDE automatically validates:
- Real-time syntax checking
- Field completion suggestions
- Error highlighting
- Warning messages

---

## Best Practices

1. **Always validate before training**
   ```bash
   okto validate train.okt
   ```

2. **Check dataset format**
   - Use `okto validate --dataset-only` to verify dataset structure

3. **Verify model compatibility**
   - Ensure model architecture matches dataset type
   - Check export format compatibility

4. **Test with small dataset first**
   - Use subset of data for initial validation
   - Verify pipeline works before full training

5. **Monitor resource usage**
   - Check available GPU memory
   - Verify disk space for checkpoints
   - Monitor training progress

---

**For more information, see:**
- [Grammar Specification](./docs/grammar.md)
- [Getting Started Guide](./docs/GETTING_STARTED.md)
- [Troubleshooting](./docs/grammar.md#troubleshooting)

