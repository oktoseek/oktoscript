# OktoScript Project Structure

Complete folder structure for the OktoScript repository.

```
oktoscript/
├── README.md                    # Main project documentation
├── LICENSE                      # Apache 2.0 License
├── .gitignore                   # Git ignore rules
├── STRUCTURE.md                 # This file
│
├── assets/                      # Visual assets
│   ├── README.md               # Assets documentation
│   ├── oktoscript_banner.png   # (Place image here)
│   └── okto_logo.png           # (Place image here)
│
├── docs/                        # Documentation
│   └── grammar.md              # Complete grammar specification
│
├── examples/                    # Example projects
│   ├── README.md               # Examples documentation
│   ├── pizzabot.okt            # Simple example file
│   └── pizzabot/               # Complete PizzaBot example
│       ├── okt.yaml            # Project configuration
│       ├── dataset/            # Training datasets
│       │   ├── train.jsonl
│       │   ├── val.jsonl
│       │   └── test.jsonl
│       ├── scripts/            # OktoScript files
│       │   └── train.okt       # Complete training script
│       ├── runs/               # Training outputs
│       │   └── pizzabot-v1/
│       │       ├── checkpoint-100/
│       │       ├── metrics.json
│       │       └── training_logs.json
│       └── export/             # Exported models
│
└── schemas/                     # JSON schemas
    └── dataset.schema.json     # Dataset validation schema
```

## File Descriptions

### Root Files
- **README.md** - Professional documentation with logos, examples, and complete project overview
- **LICENSE** - Apache 2.0 license file
- **.gitignore** - Git ignore patterns for Python, Node, training outputs, etc.

### Documentation
- **docs/grammar.md** - Complete formal grammar specification in EBNF format

### Examples
- **examples/pizzabot.okt** - Simple, minimal example
- **examples/pizzabot/** - Complete working example with all components

### Schemas
- **schemas/dataset.schema.json** - JSON Schema for validating dataset files

## Next Steps

1. Add your logo images to `assets/`:
   - `oktoscript_banner.png` (1200x300px recommended)
   - `okto_logo.png` (240x240px recommended)

2. Initialize git repository:
   ```bash
   cd oktoscript
   git init
   git add .
   git commit -m "Initial commit: OktoScript v1.0"
   ```

3. Push to GitHub:
   ```bash
   git remote add origin https://github.com/oktoseek/oktoscript.git
   git push -u origin main
   ```

## Notes

- All example code is in English as requested
- Dataset files follow JSONL format
- Training configuration follows OktoScript grammar v1.0
- Structure is ready for GitHub publication

