

## 0. Setup rapido
- Requisitos: Python 3.10+, GPU com CUDA 12.6 (ajuste o link do Torch se usar outra versao).
- Ambiente e deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2
# Se nao quiser logar no Weights & Biases:
export WANDB_DISABLED=true
```

## 1. Como os comandos de treino funcionam
- O `pretrain.py` usa Hydra e carrega `config/cfg_pretrain.yaml` por padrao. Qualquer `chave=valor` na linha de comando sobrescreve o config. Prefixo `+chave` cria chave nova.
- Estrutura geral (uma GPU):
```bash
python pretrain.py \
  arch=trm \
  data_paths="[CAMINHO_DO_DATASET]" \
  epochs=EPOCAS eval_interval=FREQUENCIA_AVALIACAO \
  lr=LR_BASE lr_warmup_steps=PASSOS_WARMUP lr_min_ratio=RAZAO_LR_FINAL \
  arch.H_cycles=H arch.L_cycles=L arch.L_layers=CAMADAS \
  +run_name="nome_da_execucao" checkpoint_path="checkpoints/minha_execucao"
```
- Multi-GPU: prefixe com `torchrun --nproc-per-node N --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1` e mantenha os mesmos overrides.
- Principais metricas: `train/*` e `test/*` incluem `lm_loss`, `q_halt_loss`, `accuracy` (token), `exact_accuracy` (puzzle inteiro) e `steps` (quantidade media de iteracoes ate halting).

## 2. Preparar dados
O pipeline espera `train/` e `test/` com `dataset.json` + npy no formato de `puzzle_dataset.py`. Para Sudoku 9x9 ja existe o builder abaixo.

### Sudoku 9x9 padrao
Arquivo: `dataset/build_sudoku_dataset.py`
```bash
# 1000 puzzles de treino, 1000 augmentations cada um
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num_aug 1000
```
Parametros uteis:
- `--output-dir`: destino do dataset convertido.
- `--subsample-size`: limita a quantidade de puzzles de treino (None usa todos).
- `--num_aug`: augmentations extras por puzzle de treino.
- `--min_difficulty`: filtra por dificuldade minima.
- `--source-repo`: repo HF a baixar (padrao sapientinc/sudoku-extreme, 9x9).

## 3. Rodando Sudoku 9x9 (baseline de referencia)
```bash
run_name="pretrain_att_sudoku_pt"
python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  data_paths_test="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +run_name=${run_name} checkpoint_path="checkpoints/${run_name}" ema=True
```
O modelo loga perda e acuracias em treino/teste. Para depurar sem W&B, mantenha `WANDB_DISABLED=true`.

## 4. Exemplo: Sudoku 6x6 e verificacao de memorizar passos
O codigo aceita qualquer tamanho de sequencia desde que o dataset siga o formato. Para 6x6 voce precisa de um dataset com `seq_len=36` (6x6) e vocab com digitos usados. Passos sugeridos:
1) Preparar dataset 6x6 no mesmo formato do `puzzle_dataset.py` (train/test com `dataset.json`, `all__inputs.npy`, `all__labels.npy` etc). Adapte `dataset/build_sudoku_dataset.py` ou gere manualmente a partir de um CSV 6x6 (36 caracteres por linha, 0 para celulas vazias) ajustando `seq_len` e `vocab_size` no metadata.
2) Rodar treino/eval apenas nesse dataset, desativando puzzle embeddings para reduzir o risco de decorar puzzles especificos (`arch.puzzle_emb_ndim=0`).
```bash
run_name="sudoku6x6_debug"
python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-6x6]" \
  data_paths_test="[data/sudoku-6x6]" \
  evaluators="[]" \
  arch.puzzle_emb_ndim=0 \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
  global_batch_size=256 \
  epochs=20000 eval_interval=2000 \
  lr=1e-4 lr_warmup_steps=2000 lr_min_ratio=0.5 \
  weight_decay=0.1 puzzle_emb_lr=0 \
  +run_name=${run_name} checkpoint_path="checkpoints/${run_name}" \
  eval_save_outputs="[preds]"
```
3) Como checar se esta decorando:
- Compare `train/exact_accuracy` vs `test/exact_accuracy`. Gap grande com perda baixa em treino sugere memoria.
- Observe `test/steps`: se cair para 1 rapido enquanto a acuracia nao sobe, o modelo pode estar dando chute repetido.
- Salve predicoes (`eval_save_outputs="[preds]"`) e abra o arquivo salvo em `checkpoints/${run_name}/step_*_all_preds.0` para ver se as solucoes repetem.
- Opcional: aumente `eval_interval` para avaliar com mais frequencia e reduzir risco de overfit invisivel.

## 5. Parametros chave de referencia
- Dados: `data_paths` (treino), `data_paths_test` (avaliacao), `evaluators` (lista, vazio para Sudoku), `global_batch_size`.
- Otimizacao: `lr`, `lr_min_ratio`, `lr_warmup_steps`, `weight_decay`, `puzzle_emb_lr`, `beta1`, `beta2`, `epochs`, `eval_interval`.
- Arquitetura: `arch.H_cycles` (iteracoes de refinamento externas), `arch.L_cycles` (iteracoes internas), `arch.L_layers` (profundidade do bloco), `arch.mlp_t` (usa MLP em vez de attn no nivel L), `arch.puzzle_emb_ndim` (0 desliga embedding por puzzle), `arch.pos_encodings` (`rope`, `learned` ou `none`).
- Outros: `ema` (liga EMA), `checkpoint_path` (onde salvar pesos/preds), `run_name`/`project_name` (nomes no W&B e pastas), `eval_save_outputs` (ex.: `[preds]` para salvar saidas de inferencia).
