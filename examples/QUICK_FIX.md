# 🔧 Correção Rápida - Erro de Parsing

## Problema: "Failed to parse file"

### Solução Rápida:

1. **Use o comando validate primeiro para ver o erro detalhado:**
```bash
okto validate scripts/train.okt
```

2. **Verifique o encoding do arquivo:**
   - No VSCode: veja no canto inferior direito → deve mostrar "UTF-8"
   - Se não for UTF-8, clique e selecione "Save with Encoding" → "UTF-8"

3. **Copie um arquivo de exemplo limpo:**
```bash
# Copie o exemplo limpo
cp oktoscript/examples/test-t5-basic-clean.okt scripts/train.okt
```

4. **Ou crie manualmente com este conteúdo mínimo:**

```okt
# okto_version: "1.2"

PROJECT "test_t5_basic"

ENV {
  accelerator: "gpu"
  min_memory: "4GB"
  install_missing: true
}

DATASET {
  train: "dataset/train.jsonl"
  validation: "dataset/val.jsonl"
}

MODEL {
  base: "google/t5-small"
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

### ⚠️ Problemas Comuns:

1. **Bloco de Notas do Windows** adiciona BOM (Byte Order Mark)
   - **Solução:** Use VSCode ou Notepad++

2. **Caracteres especiais** em comentários ou strings
   - **Solução:** Use apenas ASCII ou UTF-8 válido

3. **Aspas curvas** `"` ou `"` ao invés de retas `"`
   - **Solução:** Use sempre aspas retas

4. **Espaços invisíveis** ou caracteres de controle
   - **Solução:** Re-digite o arquivo ou use um editor que mostre caracteres invisíveis

### ✅ Teste Rápido:

```bash
# 1. Validar
okto validate scripts/train.okt

# 2. Se validar, treinar
okto train scripts/train.okt
```




