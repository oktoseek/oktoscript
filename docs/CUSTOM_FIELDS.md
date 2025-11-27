# Campos Customizados no Dataset (v1.2+)

## Visão Geral

A partir da versão 1.2, o OktoScript permite definir campos customizados para input e output no bloco `DATASET`. Além disso, você pode especificar **campos de contexto** que serão automaticamente incluídos no prompt durante o treinamento. Isso oferece flexibilidade total para trabalhar com datasets complexos que incluem informações contextuais (como menu, drinks, promoções, etc.).

## Sintaxe

```okt
DATASET {
    train: "dataset/train.jsonl"
    validation: "dataset/val.jsonl"
    format: "jsonl"
    
    # Campos customizados (opcional)
    input_field: "input"      # Nome da coluna de entrada
    output_field: "target"    # Nome da coluna de saída (ou use target_field)
    
    # Campos de contexto (opcional) - incluídos automaticamente no prompt
    context_fields: ["menu", "drinks", "promotions"]
}
```

## Resolução Automática de Campos

Se você **não especificar** `input_field` e `output_field`, o OktoEngine tentará encontrar os campos automaticamente na seguinte ordem:

### Para modelos Seq2Seq (T5, BART, etc.):

1. **`input` + `output`** (padrão mais comum)
2. **`input` + `target`** (alternativa comum)
3. **`text`** (campo único, usado para ambos)
4. **Primeiro campo string encontrado** (fallback)

### Para modelos Causal (GPT, etc.):

1. **`input` + `output`** (concatenados)
2. **`input` + `target`** (concatenados)
3. **`text`** (campo único)
4. **Primeiro campo string encontrado** (fallback)

## Exemplos de Uso

### Exemplo 1: Dataset com `input` e `target`

```okt
DATASET {
    train: "dataset/train.jsonl"
    format: "jsonl"
    input_field: "input"
    output_field: "target"
}
```

**Dataset JSONL:**
```jsonl
{"input": "User: Olá", "target": "Assistant: Olá! Como posso ajudar?"}
{"input": "User: Tudo bem?", "target": "Assistant: Sim, tudo ótimo!"}
```

### Exemplo 2: Dataset com campos diferentes

```okt
DATASET {
    train: "dataset/conversations.jsonl"
    format: "jsonl"
    input_field: "question"
    output_field: "answer"
}
```

**Dataset JSONL:**
```jsonl
{"question": "Qual é a capital do Brasil?", "answer": "Brasília"}
{"question": "Quem descobriu o Brasil?", "answer": "Pedro Álvares Cabral"}
```

### Exemplo 3: Dataset com nomes em português

```okt
DATASET {
    train: "dataset/treino.jsonl"
    format: "jsonl"
    input_field: "entrada"
    output_field: "saida"
}
```

**Dataset JSONL:**
```jsonl
{"entrada": "Traduza: Hello", "saida": "Olá"}
{"entrada": "Traduza: Goodbye", "saida": "Adeus"}
```

### Exemplo 4: Sem especificar campos (auto-detecção)

```okt
DATASET {
    train: "dataset/train.jsonl"
    format: "jsonl"
    # input_field e output_field não especificados
    # O engine tentará encontrar automaticamente
}
```

O engine tentará:
- `input` + `output` → se não encontrar
- `input` + `target` → se não encontrar
- `text` → se não encontrar
- Primeiro campo string → fallback

## Compatibilidade

### Retrocompatibilidade

Scripts antigos continuam funcionando sem modificação:

```okt
# Script v1.0/v1.1 - funciona perfeitamente
DATASET {
    train: "dataset/train.jsonl"
    validation: "dataset/val.jsonl"
}
```

O engine detectará automaticamente `input`/`output` ou `input`/`target`.

### Aliases Suportados

- `output_field` e `target_field` são equivalentes
- Ambos podem ser usados para definir o campo de saída

```okt
DATASET {
    train: "dataset/train.jsonl"
    input_field: "input"
    output_field: "target"   # ou target_field: "target"
}
```

## Casos de Uso

### 1. Datasets de Terceiros

Quando você usa datasets de repositórios públicos que podem ter nomes de colunas diferentes:

```okt
DATASET {
    train: "datasets/alpaca_pt.jsonl"
    input_field: "instruction"
    output_field: "response"
}
```

### 2. Migração de Formatos

Ao migrar de outros frameworks que usam convenções diferentes:

```okt
DATASET {
    train: "dataset/old_format.jsonl"
    input_field: "prompt"
    output_field: "completion"
}
```

### 3. Datasets Multilíngues

Para datasets que misturam idiomas nos nomes das colunas:

```okt
DATASET {
    train: "dataset/mixed.jsonl"
    input_field: "entrada"
    output_field: "saida"
}
```

## Validação

O OktoEngine valida que:

- Os campos especificados existem no dataset
- Os campos contêm dados válidos (strings)
- O formato do dataset é compatível

## Dicas

1. **Use campos customizados quando necessário**: Se seu dataset já usa `input`/`output` ou `input`/`target`, não precisa especificar.

2. **Teste primeiro**: Use `okto validate` para verificar se os campos estão corretos antes de treinar.

3. **Consistência**: Mantenha os mesmos nomes de campos em train, validation e test.

4. **Documentação**: Documente os nomes de campos customizados no seu projeto para facilitar colaboração.

## Troubleshooting

### Erro: "Field 'X' not found in dataset"

**Causa**: O campo especificado não existe no dataset.

**Solução**: 
- Verifique os nomes das colunas no seu dataset
- Use `okto validate` para ver quais campos foram detectados
- Remova `input_field`/`output_field` para usar auto-detecção

### Erro: "No input/output fields found"

**Causa**: O engine não conseguiu encontrar campos válidos.

**Solução**:
- Especifique explicitamente `input_field` e `output_field`
- Verifique se o dataset tem pelo menos um campo string

### Dataset funciona sem especificar campos, mas falha com campos customizados

**Causa**: Nome do campo incorreto ou com espaços/caracteres especiais.

**Solução**:
- Use exatamente o nome da coluna como aparece no JSON
- Evite espaços ou caracteres especiais nos nomes das colunas

## Exemplo Completo

```okt
# okto_version: "1.2"
PROJECT "custom_fields_example"

DATASET {
    train: "dataset/train.jsonl"
    validation: "dataset/val.jsonl"
    format: "jsonl"
    type: "chat"
    
    # Campos customizados
    input_field: "user_message"
    output_field: "assistant_response"
}

MODEL {
    base: "t5-small"
    device: "auto"
}

TRAIN {
    epochs: 3
    batch_size: 8
    learning_rate: 0.0001
}

EXPORT {
    format: ["okm"]
    path: "export/"
}
```

---

**Versão**: 1.2+  
**Última atualização**: 2024

