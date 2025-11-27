# Campos de Contexto (Context Fields) - v1.2+

## Visão Geral

Campos de contexto são campos adicionais no seu dataset que contêm informações que devem ser incluídas automaticamente no prompt durante o treinamento, mas não são a entrada principal nem a saída esperada.

## Casos de Uso

### 1. Chatbots com Contexto Dinâmico

Para chatbots que precisam de informações contextuais (menu, drinks, promoções, horários, etc.):

```okt
DATASET {
    train: "dataset/pizzaria.jsonl"
    format: "jsonl"
    input_field: "input"
    output_field: "target"
    context_fields: ["menu", "drinks", "promotions"]
}
```

**Dataset JSONL:**
```jsonl
{"input": "What pizzas do you have?", "target": "We have Margherita, Pepperoni, and Four Cheese.", "menu": "Margherita: $34, Pepperoni: $39, Four Cheese: $45", "drinks": "Coke, Sprite, Water"}
{"input": "Any promotions?", "target": "Yes! Buy 2 get 1 free on Tuesdays.", "menu": "Margherita: $34, Pepperoni: $39", "promotions": "Buy 2 get 1 free on Tuesdays"}
```

**Prompt gerado automaticamente:**
```
menu: Margherita: $34, Pepperoni: $39, Four Cheese: $45 | drinks: Coke, Sprite, Water | What pizzas do you have?
```

### 2. Question Answering com Documentos

Para QA que precisa de contexto do documento:

```okt
DATASET {
    train: "dataset/qa.jsonl"
    format: "jsonl"
    input_field: "question"
    output_field: "answer"
    context_fields: ["document", "section"]
}
```

**Dataset JSONL:**
```jsonl
{"question": "What is the capital?", "answer": "Brasília", "document": "Geography of Brazil", "section": "Administrative divisions"}
```

### 3. Tradução com Contexto

Para tradução que precisa de contexto adicional:

```okt
DATASET {
    train: "dataset/translation.jsonl"
    format: "jsonl"
    input_field: "source"
    output_field: "target"
    context_fields: ["domain", "style"]
}
```

## Como Funciona

### Formato do Prompt

Os campos de contexto são incluídos **antes** do `input_field` no formato:

```
{context_field_1}: {value_1} | {context_field_2}: {value_2} | {input_field}
```

### Ordem dos Campos

Os campos são incluídos na **ordem especificada** em `context_fields`:

```okt
context_fields: ["menu", "drinks", "promotions"]
```

Resultado: `menu: ... | drinks: ... | promotions: ... | input`

### Campos Vazios

Campos vazios são **automaticamente ignorados**:

```jsonl
{"input": "Hello", "target": "Hi", "menu": "", "drinks": "Coke"}
```

Resultado: `drinks: Coke | Hello` (menu vazio foi ignorado)

## Sintaxe

```okt
DATASET {
    train: "dataset/train.jsonl"
    validation: "dataset/val.jsonl"
    format: "jsonl"
    
    # Campos principais
    input_field: "input"
    output_field: "target"
    
    # Campos de contexto (opcional)
    context_fields: ["menu", "drinks", "promotions"]
}
```

## Exemplo Completo

```okt
# okto_version: "1.2"
PROJECT "pizzaria_chatbot"

DATASET {
    train: "dataset/pizzaria_train.jsonl"
    validation: "dataset/pizzaria_val.jsonl"
    format: "jsonl"
    type: "chat"
    
    input_field: "input"
    output_field: "target"
    context_fields: ["menu", "drinks", "promotions", "hours"]
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

**Dataset de exemplo:**
```jsonl
{"input": "What pizzas do you have?", "target": "We have Margherita ($34), Pepperoni ($39), and Four Cheese ($45).", "menu": "Margherita: $34, Pepperoni: $39, Four Cheese: $45", "drinks": "Coke, Sprite, Water", "promotions": "Buy 2 get 1 free on Tuesdays", "hours": "Open 11am-11pm daily"}
{"input": "What time do you close?", "target": "We close at 11pm daily.", "menu": "Margherita: $34, Pepperoni: $39", "hours": "Open 11am-11pm daily"}
```

## Dicas e Boas Práticas

### ✅ Faça

1. **Use nomes descritivos**: `menu`, `drinks`, `promotions` são melhores que `ctx1`, `ctx2`
2. **Mantenha contexto relevante**: Apenas campos que realmente ajudam o modelo
3. **Ordene por importância**: Campos mais importantes primeiro
4. **Use consistentemente**: Mesmos campos em train, validation e test

### ❌ Evite

1. **Muitos campos**: Mais de 5-6 campos pode confundir o modelo
2. **Campos muito longos**: Contexto muito extenso pode ultrapassar o limite de tokens
3. **Informação redundante**: Não repita informação já no input
4. **Campos não relacionados**: Apenas contexto relevante para a tarefa

## Limitações

- Campos de contexto são incluídos no prompt, então contam para o limite de tokens
- A ordem dos campos importa - coloque os mais importantes primeiro
- Campos vazios são ignorados automaticamente
- Funciona tanto para modelos Seq2Seq (T5, BART) quanto Causal (GPT)

## Troubleshooting

### O contexto não está aparecendo no prompt

**Causa**: Nome do campo incorreto ou campo não existe no dataset.

**Solução**: 
- Verifique os nomes dos campos no dataset
- Use `okto validate` para verificar a configuração
- Certifique-se de que os campos existem em todos os exemplos

### Prompt muito longo

**Causa**: Muitos campos de contexto ou campos muito longos.

**Solução**:
- Reduza o número de campos de contexto
- Encurte o conteúdo dos campos
- Aumente `max_length` no tokenizer (se necessário)

---

**Versão**: 1.2+  
**Última atualização**: 2024




