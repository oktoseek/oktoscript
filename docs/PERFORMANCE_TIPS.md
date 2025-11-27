# Dicas de Performance para Treinamento

## Configuração de Logs

### logging_steps

Controla com que frequência os logs são exibidos durante o treinamento:

```okt
TRAIN {
    epochs: 3
    batch_size: 8
    learning_rate: 0.0001
    logging_steps: 5    # Log a cada 5 steps (mais frequente)
    # logging_steps: 20  # Log a cada 20 steps (menos frequente)
}
```

**Valores recomendados:**
- **Datasets pequenos (< 1000 exemplos)**: `logging_steps: 5` ou `10`
- **Datasets médios (1000-10000)**: `logging_steps: 10` ou `20`
- **Datasets grandes (> 10000)**: `logging_steps: 50` ou `100`

### save_steps

Controla com que frequência os checkpoints são salvos:

```okt
TRAIN {
    epochs: 3
    batch_size: 8
    learning_rate: 0.0001
    save_steps: 100     # Salva checkpoint a cada 100 steps
    # save_steps: 500    # Salva checkpoint a cada 500 steps (padrão)
}
```

**Dica**: Para datasets pequenos, use `save_steps` menor para não perder progresso.

## Otimização de Performance

### 1. Use GPU quando disponível

```okt
ENV {
    accelerator: "gpu"
    precision: "fp16"    # Usa menos memória e é mais rápido
}

MODEL {
    base: "t5-small"
    device: "cuda"       # Força uso de GPU
}
```

**Problema comum**: Se você tem GPU mas vê "No CUDA", instale:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Aumente o batch_size (se tiver memória)

```okt
TRAIN {
    epochs: 3
    batch_size: 16       # Aumente de 8 para 16 ou 32 (se tiver memória)
    learning_rate: 0.0001
}
```

**Trade-off**: 
- Batch maior = treinamento mais rápido, mas usa mais memória
- Batch menor = treinamento mais lento, mas usa menos memória

### 3. Use gradient_accumulation para simular batch maior

Se não tiver memória para batch grande, use gradient accumulation:

```okt
TRAIN {
    epochs: 3
    batch_size: 8
    gradient_accumulation: 4    # Efetivamente batch_size = 8 * 4 = 32
    learning_rate: 0.0001
}
```

### 4. Reduza o tamanho do input (se possível)

Se seus inputs são muito longos (como JSON de menu embutido), considere:

- **Usar context_fields**: Mova informações longas para campos de contexto
- **Truncar inputs**: O tokenizer já faz isso (max_length: 512), mas inputs menores são mais rápidos

### 5. Use modelos menores para testes

Para desenvolvimento/testes rápidos:

```okt
MODEL {
    base: "t5-small"      # Mais rápido
    # base: "t5-base"     # Mais lento, mas melhor qualidade
}
```

### 6. Reduza epochs para testes

```okt
TRAIN {
    epochs: 1             # Para testes rápidos
    # epochs: 3          # Para treinamento real
    batch_size: 8
}
```

## Análise do Seu Caso

Com base no seu dataset (582 exemplos, inputs longos com Menu JSON):

### Por que está lento?

1. **Sem CUDA**: Você está usando CPU, que é ~10-50x mais lento que GPU
2. **Batch size pequeno (8)**: Com 582 exemplos e batch 8, são ~73 steps por epoch
3. **Inputs longos**: O Menu JSON embutido aumenta o tempo de processamento
4. **Modelo T5**: T5-small é relativamente pesado para CPU

### Soluções Imediatas:

1. **Instalar CUDA** (se tiver GPU NVIDIA):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Aumentar batch_size** (se tiver RAM):
   ```okt
   TRAIN {
       epochs: 3
       batch_size: 16    # Dobrar o batch_size
       logging_steps: 5  # Logs mais frequentes
   }
   ```

3. **Usar context_fields** (recomendado):
   ```okt
   DATASET {
       train: "dataset/train.jsonl"
       input_field: "input"
       output_field: "target"
       context_fields: ["menu"]  # Move Menu para contexto
   }
   ```
   
   E no dataset, separe o Menu:
   ```jsonl
   {"input": "What pizzas do you have?", "target": "...", "menu": "{\"Margherita\":34,...}"}
   ```

4. **Reduzir epochs para testes**:
   ```okt
   TRAIN {
       epochs: 1         # Teste rápido
       batch_size: 16
       logging_steps: 5
   }
   ```

## Tempo Esperado

### CPU (seu caso atual):
- **582 exemplos, batch 8, 3 epochs**: ~30-60 minutos
- **Com batch 16**: ~15-30 minutos

### GPU (com CUDA):
- **582 exemplos, batch 8, 3 epochs**: ~3-5 minutos
- **Com batch 16**: ~2-3 minutos

## Exemplo Otimizado

```okt
# okto_version: "1.2"
PROJECT "pizzaria_optimized"

ENV {
    accelerator: "gpu"
    precision: "fp16"
    backend: "oktoseek"
}

DATASET {
    train: "dataset/train.jsonl"
    validation: "dataset/val.jsonl"
    input_field: "input"
    output_field: "target"
    context_fields: ["menu"]  # Menu separado como contexto
}

MODEL {
    base: "t5-small"
    device: "cuda"
}

TRAIN {
    epochs: 3
    batch_size: 16           # Aumentado
    learning_rate: 0.0001
    logging_steps: 5         # Logs mais frequentes
    save_steps: 50           # Salva checkpoints mais frequentemente
}

EXPORT {
    format: ["okm"]
    path: "export/"
}
```

---

**Última atualização**: 2024




