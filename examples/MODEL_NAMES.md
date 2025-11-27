# Modelos Válidos no HuggingFace

## Modelos T5

| Nome no HuggingFace | Tamanho | Descrição |
|---------------------|---------|-----------|
| `t5-small` | 60M | T5 pequeno - rápido para testes |
| `t5-base` | 220M | T5 base - bom equilíbrio |
| `t5-large` | 770M | T5 grande - melhor qualidade |
| `google/flan-t5-small` | 60M | Flan-T5 pequeno |
| `google/flan-t5-base` | 220M | Flan-T5 base |
| `google/flan-t5-large` | 780M | Flan-T5 grande |

## Modelos GPT (Causal LM)

| Nome no HuggingFace | Tamanho | Descrição |
|---------------------|---------|-----------|
| `gpt2` | 124M | GPT-2 padrão |
| `distilgpt2` | 82M | GPT-2 destilado - mais rápido |
| `microsoft/DialoGPT-small` | 117M | DialoGPT pequeno |
| `EleutherAI/gpt-neo-125M` | 125M | GPT-Neo pequeno |
| `facebook/opt-125m` | 125M | OPT pequeno |

## ⚠️ Erro Comum

**❌ ERRADO:**
```okt
MODEL {
  base: "google/t5-small"  # ❌ Não existe!
}
```

**✅ CORRETO:**
```okt
MODEL {
  base: "t5-small"  # ✅ Correto!
}
```

## Como Verificar se um Modelo Existe

1. Acesse: https://huggingface.co/models
2. Busque pelo nome do modelo
3. Verifique o nome exato na URL ou na página do modelo

## Exemplos de Uso

### T5 para Tradução/Sumarização
```okt
MODEL {
  base: "t5-small"
}
```

### Flan-T5 para Chat/Instruções
```okt
MODEL {
  base: "google/flan-t5-base"
}
```

### GPT-2 para Geração de Texto
```okt
MODEL {
  base: "gpt2"
}
```




