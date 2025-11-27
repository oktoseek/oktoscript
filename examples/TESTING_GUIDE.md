# Guia de Testes - OktoScript v1.2

Este guia lista todos os scripts de teste disponíveis e como usá-los para validar diferentes funcionalidades do OktoScript.

## 📋 Scripts de Teste Disponíveis

### 1. `test-t5-basic.okt` - Treinamento Básico
**Objetivo:** Testar treinamento simples sem blocos avançados

**Modelo:** `google/t5-small`

**Blocos usados:**
- PROJECT
- ENV
- DATASET
- MODEL
- TRAIN
- EXPORT

**Como testar:**
```bash
okto validate examples/test-t5-basic.okt
okto train examples/test-t5-basic.okt
```

**O que verificar:**
- ✅ Treinamento inicia sem erros
- ✅ Modelo é salvo em `runs/test_t5_basic/`
- ✅ Export funciona para `okm` e `safetensors`

---

### 2. `test-t5-monitor.okt` - Monitoramento de Métricas
**Objetivo:** Testar bloco MONITOR com tracking completo

**Modelo:** `google/t5-small`

**Blocos usados:**
- MONITOR (completo)
- Métricas: loss, val_loss, accuracy, perplexity, gpu_usage, ram_usage, throughput, latency

**Como testar:**
```bash
okto validate examples/test-t5-monitor.okt
okto train examples/test-t5-monitor.okt
okto logs test_t5_monitor
```

**O que verificar:**
- ✅ Métricas são coletadas durante treinamento
- ✅ Arquivo `logs/training_monitor.log` é criado
- ✅ Notificações são geradas quando condições são atendidas

---

### 3. `test-t5-control.okt` - Controle e Decisões
**Objetivo:** Testar bloco CONTROL com lógica condicional

**Modelo:** `google/t5-small`

**Blocos usados:**
- CONTROL (completo)
- Eventos: on_step_end, on_epoch_end
- Diretivas: IF, WHEN, EVERY, SET, STOP_TRAINING, DECREASE, SAVE, LOG

**Como testar:**
```bash
okto validate examples/test-t5-control.okt
okto train examples/test-t5-control.okt
okto logs test_t5_control
```

**O que verificar:**
- ✅ Logs são gerados em cada step/epoch
- ✅ Learning rate é ajustado automaticamente quando loss > 2.0
- ✅ Treinamento para quando val_loss > 2.5
- ✅ Checkpoints são salvos a cada 500 steps
- ✅ Arquivo `control_decisions.json` é criado em `runs/test_t5_control/`

---

### 4. `test-flan-t5-complete.okt` - Todos os Blocos
**Objetivo:** Testar todos os blocos avançados juntos

**Modelo:** `google/flan-t5-base`

**Blocos usados:**
- MONITOR (completo)
- CONTROL (completo com lógica aninhada)
- STABILITY
- EXPORT

**Como testar:**
```bash
okto validate examples/test-flan-t5-complete.okt
okto train examples/test-flan-t5-complete.okt
okto logs test_flan_t5_complete
```

**O que verificar:**
- ✅ Todos os blocos funcionam juntos
- ✅ Lógica aninhada no CONTROL funciona (IF dentro de on_epoch_end)
- ✅ STABILITY previne NaN e divergência
- ✅ Métricas completas são coletadas

---

### 5. `test-flan-t5-inference.okt` - Inferência com Governança
**Objetivo:** Testar inferência com BEHAVIOR, GUARD e INFERENCE

**Modelo:** `google/flan-t5-base`

**Blocos usados:**
- BEHAVIOR (personality, language, avoid)
- GUARD (prevent, detect_using, on_violation)
- INFERENCE (mode, format, params, CONTROL aninhado)

**Como testar:**
```bash
# Treinar primeiro
okto train examples/test-flan-t5-inference.okt

# Testar inferência
okto infer --model export/test_flan_t5_inference --text "Olá, como você está?"

# Testar chat interativo
okto chat --model export/test_flan_t5_inference
```

**O que verificar:**
- ✅ Modelo respeita BEHAVIOR (personality, language)
- ✅ GUARD bloqueia conteúdo tóxico/inadequado
- ✅ INFERENCE usa formato correto
- ✅ CONTROL dentro de INFERENCE funciona (RETRY, REGENERATE)

---

### 6. `test-t5-explorer.okt` - AutoML Básico
**Objetivo:** Testar bloco EXPLORER para busca de hiperparâmetros

**Modelo:** `google/t5-small`

**Blocos usados:**
- EXPLORER (try, max_tests, pick_best_by)
- MONITOR

**Como testar:**
```bash
okto validate examples/test-t5-explorer.okt
okto train examples/test-t5-explorer.okt
```

**O que verificar:**
- ✅ Múltiplas combinações de hiperparâmetros são testadas
- ✅ Melhor modelo é selecionado por val_loss
- ✅ Logs mostram resultados de cada teste

---

## 🧪 Sequência Recomendada de Testes

### Fase 1: Testes Básicos
1. `test-t5-basic.okt` - Validar pipeline básico
2. `test-t5-monitor.okt` - Validar monitoramento

### Fase 2: Testes de Controle
3. `test-t5-control.okt` - Validar decisões automáticas
4. `test-flan-t5-complete.okt` - Validar integração completa

### Fase 3: Testes Avançados
5. `test-flan-t5-inference.okt` - Validar inferência governada
6. `test-t5-explorer.okt` - Validar AutoML

---

## 📊 Checklist de Validação

Para cada teste, verifique:

- [ ] Script valida sem erros (`okto validate`)
- [ ] Treinamento inicia corretamente
- [ ] Blocos específicos funcionam como esperado
- [ ] Logs são gerados corretamente
- [ ] Export funciona para formato especificado
- [ ] Arquivos são salvos nos locais corretos

---

## 🔍 Comandos Úteis

```bash
# Validar script
okto validate examples/test-t5-basic.okt

# Treinar
okto train examples/test-t5-basic.okt

# Ver logs
okto logs test_t5_basic

# Inferência
okto infer --model export/test_t5_basic --text "Hello"

# Chat interativo
okto chat --model export/test_t5_basic

# Ver conteúdo do script
okto show examples/test-t5-basic.okt
```

---

## 📝 Notas

- Todos os testes usam `dataset/train.jsonl` e `dataset/val.jsonl`
- Certifique-se de ter dados de teste antes de executar
- Modelos T5 são menores e mais rápidos para testes
- Modelos Flan-T5 são melhores para inferência e chat
- Ajuste `batch_size` e `epochs` conforme sua GPU

---

**Boa sorte com os testes! 🚀**




