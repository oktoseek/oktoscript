# Troubleshooting - Problemas Comuns

## Erro: "Failed to parse file"

### Possíveis Causas:

1. **Encoding do arquivo**
   - O arquivo deve estar em **UTF-8 sem BOM**
   - Evite salvar no Bloco de Notas do Windows (pode adicionar BOM)
   - Use VSCode, Notepad++, ou outro editor que suporte UTF-8

2. **Caracteres invisíveis**
   - Copiar/colar pode adicionar caracteres invisíveis
   - Re-digite o arquivo ou use um editor que mostre caracteres invisíveis

3. **Problemas com comentários**
   - Comentários devem começar com `#` no início da linha
   - Evite caracteres especiais em comentários

4. **Aspas incorretas**
   - Use aspas retas `"` não aspas curvas `"` ou `"`
   - Verifique se todas as aspas estão fechadas

### Soluções:

#### 1. Validar o arquivo primeiro:
```bash
okto validate scripts/train.okt
```

Isso mostrará erros detalhados.

#### 2. Verificar encoding no VSCode:
- Abra o arquivo no VSCode
- Veja no canto inferior direito: deve mostrar "UTF-8"
- Se mostrar outro encoding, clique e selecione "Save with Encoding" → "UTF-8"

#### 3. Criar arquivo limpo:
```bash
# Copie o conteúdo do exemplo
cp oktoscript/examples/test-t5-basic.okt scripts/train.okt

# Ou crie manualmente
```

#### 4. Verificar sintaxe básica:
- Todas as strings devem estar entre aspas: `"valor"`
- Arrays devem usar colchetes: `["okm", "safetensors"]`
- Blocos devem ter chaves: `{ ... }`
- Não use vírgulas no final de arrays ou objetos

### Exemplo de arquivo correto:

```okt
# okto_version: "1.2"

PROJECT "test_t5_basic"
DESCRIPTION "Teste basico"

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

### Checklist:

- [ ] Arquivo está em UTF-8
- [ ] Todas as aspas estão fechadas
- [ ] Não há caracteres especiais invisíveis
- [ ] Sintaxe está correta (chaves, colchetes, etc.)
- [ ] `okto validate` passa sem erros

### Se ainda não funcionar:

1. Execute com `--debug` (se disponível):
```bash
okto validate scripts/train.okt --debug
```

2. Verifique o conteúdo do arquivo:
```bash
okto show scripts/train.okt
```

3. Compare com um exemplo que funciona:
```bash
okto validate oktoscript/examples/test-t5-basic.okt
```




