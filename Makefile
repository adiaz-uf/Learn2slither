PYTHON  ?= python3
VENV_PY := .venv/bin/python
PIP     := .venv/bin/pip
# Subject-mandated entrypoint. The wrapper picks the venv python when
# present and forwards all arguments to src/AI_Model/main.py.
SNAKE   := ./snake
TUNE    := src/AI_Model/tune.py

SESSIONS ?= 100
BOARD    ?= 10
FPS      ?= 10
# Optuna defaults
TRIALS               ?= 50
EPISODES_PER_TRIAL   ?= 5000
# Pruning controls (see `tune.py --help` for details).
# PRUNER          : percentile | median | none
# PRUNE_PCT       : only used by 'percentile' (lower = more lenient)
# PATIENCE        : 0 disables PatientPruner wrapper
# N_WARMUP_STEPS  : episodes a trial must run before pruning kicks in
PRUNER               ?= percentile
PRUNE_PCT            ?= 25
PATIENCE             ?= 2
N_WARMUP_STEPS       ?= 1500
# Training mode: 'single' (one model per board size, default) or
# 'multi' (one portable model for 10/14/18 — bonus part).
MODE     ?= single
# Default model path. For single it resolves under models/<size>x<size>/.
# For multi it lives under models/multi/. Override with MODEL=<path> to
# load a specific checkpoint.
ifeq ($(MODE),multi)
MODEL    ?= models/multi/best_snake_multi_10000ep.keras
else
MODEL    ?= models/$(BOARD)x$(BOARD)/best_snake_$(BOARD)x$(BOARD)_3000ep.keras
endif
SAVE     ?=

.PHONY: all help create-venv install-deps setup \
	train train-visual train-debug train-continue \
	train-multi train-multi-debug train-multi-continue \
	eval eval-visual eval-step \
	eval-multi eval-multi-visual \
	models models-10 models-14 models-18 models-multi \
	tune tune-multi tune-fast tune-inspect tune-inspect-multi

all: setup

help:
	@echo "==========================================================="
	@echo "  Learn2Slither — make targets"
	@echo "==========================================================="
	@echo ""
	@echo "  Setup:"
	@echo "    make setup                                -> create venv + install requirements.txt"
	@echo ""
	@echo "  Training (single mode = fixed --board-size every episode):"
	@echo "    make train          SESSIONS=10000 BOARD=10  -> headless training"
	@echo "    make train-visual   SESSIONS=100   BOARD=14  -> training with pygame window"
	@echo "    make train-debug    SESSIONS=50    BOARD=10  -> verbose first-2-episode debug"
	@echo "    make train-continue SESSIONS=5000  BOARD=10  -> resume from MODEL"
	@echo ""
	@echo "  Training (multi mode = uniform 10/14/18 sampling per episode, bonus):"
	@echo "    make train-multi          SESSIONS=10000     -> headless multi-size training"
	@echo "    make train-multi-debug    SESSIONS=50        -> multi training with debug"
	@echo "    make train-multi-continue SESSIONS=5000      -> resume multi from MODEL"
	@echo ""
	@echo "  Evaluation (no learning, model plays):"
	@echo "    make eval               BOARD=10 SESSIONS=30 MODEL=path  -> terminal-only"
	@echo "    make eval-visual        BOARD=14              MODEL=path -> full UI lobby + game"
	@echo "    make eval-step          BOARD=10              MODEL=path -> step-by-step terminal"
	@echo "    make eval-multi-visual                        MODEL=path -> UI with 3-size selector"
	@echo ""
	@echo "  Submission models (subject requirement: 1/10/100 sessions):"
	@echo "    make models                              -> 1/10/100-session models for 10/14/18"
	@echo "    make models-10                           -> only 10x10 reference models"
	@echo "    make models-14                           -> only 14x14 reference models"
	@echo "    make models-18                           -> only 18x18 reference models"
	@echo "    make models-multi                        -> 1/10/100-session multi models (bonus)"
	@echo ""
	@echo "  Hyperparameter tuning (Optuna, lenient pruning by default):"
	@echo "    make tune              TRIALS=30 BOARD=10  -> single-mode tune (10x10 specialist)"
	@echo "    make tune-multi        TRIALS=30           -> multi-mode tune (33/33/33)"
	@echo "    make tune-fast         TRIALS=10           -> 500 ep/trial smoke check"
	@echo "    make tune-inspect       BOARD=10           -> show best params + importance + zone"
	@echo "    make tune-inspect-multi                    -> same for multi study"
	@echo ""
	@echo "    Trial 0 is always seeded with a known-good baseline (lr=5e-4, gamma=0.98,"
	@echo "    NO_EAT=-0.3, [64,64]) so the first result is sensible, not random noise."
	@echo "    Override pruning:    PRUNER=none|median|percentile  PRUNE_PCT=25"
	@echo "                          PATIENCE=2          N_WARMUP_STEPS=1500"
	@echo "    Resume study:         re-run 'make tune' — appends to the existing SQLite."
	@echo "    Restart from scratch: rm optuna_studies/<study>.db"
	@echo ""
	@echo "  Variables (current values):"
	@echo "    SESSIONS=$(SESSIONS)  BOARD=$(BOARD)  FPS=$(FPS)  MODE=$(MODE)"
	@echo "    TRIALS=$(TRIALS)  EPISODES_PER_TRIAL=$(EPISODES_PER_TRIAL)"
	@echo "    PRUNER=$(PRUNER)  PRUNE_PCT=$(PRUNE_PCT)  PATIENCE=$(PATIENCE)  N_WARMUP_STEPS=$(N_WARMUP_STEPS)"
	@echo "    MODEL=$(MODEL)"
	@echo ""
	@echo "  Tip: 'make tune-inspect' is read-only and works while a 'make tune' run is in progress."

create-venv:
	$(PYTHON) -m venv .venv
	@echo "Virtual environment created."

install-deps:
	$(PIP) install -r requirements.txt

setup: create-venv install-deps

# ---------------------------------------------------------------------------
# Training targets — single mode (default)
# Models are saved automatically to models/<BOARD>x<BOARD>/ by main.py.
# Use SAVE=<path> to override the output path.
# ---------------------------------------------------------------------------

train:
	$(SNAKE) --sessions $(SESSIONS) --board-size $(BOARD) \
		--training-mode single --visual off \
		$(if $(SAVE),--save $(SAVE),)

train-visual:
	$(SNAKE) --sessions $(SESSIONS) --board-size $(BOARD) \
		--training-mode single --visual on --fps $(FPS) \
		$(if $(SAVE),--save $(SAVE),)

train-debug:
	$(SNAKE) --sessions $(SESSIONS) --board-size $(BOARD) \
		--training-mode single --visual off --debug \
		$(if $(SAVE),--save $(SAVE),)

train-continue:
	$(SNAKE) --sessions $(SESSIONS) --board-size $(BOARD) \
		--training-mode single --visual off --load $(MODEL) \
		$(if $(SAVE),--save $(SAVE),)

# ---------------------------------------------------------------------------
# Training targets — multi mode (bonus: size-portable model)
# Each episode samples a board size uniformly from [10, 14, 18] (33/33/33).
# Models are saved under models/multi/. --board-size is irrelevant in
# this mode (overridden per episode).
# ---------------------------------------------------------------------------

train-multi:
	$(SNAKE) --sessions $(SESSIONS) \
		--training-mode multi --visual off \
		$(if $(SAVE),--save $(SAVE),)

train-multi-debug:
	$(SNAKE) --sessions $(SESSIONS) \
		--training-mode multi --visual off --debug \
		$(if $(SAVE),--save $(SAVE),)

train-multi-continue:
	$(SNAKE) --sessions $(SESSIONS) \
		--training-mode multi --visual off --load $(MODEL) \
		$(if $(SAVE),--save $(SAVE),)

# ---------------------------------------------------------------------------
# Evaluation targets — single mode
# Board size must match the model's training size; main.py reads it from
# the *_metadata.json and overrides --board-size if they disagree.
# ---------------------------------------------------------------------------

eval:
	$(SNAKE) --dontlearn --load $(MODEL) --sessions $(SESSIONS) \
		--board-size $(BOARD) --visual off

eval-visual:
	$(SNAKE) --dontlearn --load $(MODEL) --sessions $(SESSIONS) \
		--board-size $(BOARD) --visual on --fps $(FPS)

eval-step:
	$(SNAKE) --dontlearn --load $(MODEL) --sessions $(SESSIONS) \
		--board-size $(BOARD) --visual off --step-by-step

# ---------------------------------------------------------------------------
# Evaluation targets — multi mode
# eval-multi runs without UI on a specific BOARD (10/14/18).
# eval-multi-visual opens the UI lobby with the 3-size selector enabled.
# ---------------------------------------------------------------------------

eval-multi:
	$(SNAKE) --dontlearn --load $(MODEL) --sessions $(SESSIONS) \
		--board-size $(BOARD) --visual off

eval-multi-visual:
	$(SNAKE) --dontlearn --load $(MODEL) --sessions $(SESSIONS) \
		--visual on --fps $(FPS)

# ---------------------------------------------------------------------------
# Model generation targets
# Creates reference models for each board size.
# Run 'make setup' first.
# ---------------------------------------------------------------------------

models-10:
	$(SNAKE) --sessions 1   --board-size 10 --training-mode single \
		--save models/10x10/snake_10x10_1sess.keras
	$(SNAKE) --sessions 10  --board-size 10 --training-mode single \
		--save models/10x10/snake_10x10_10sess.keras
	$(SNAKE) --sessions 100 --board-size 10 --training-mode single \
		--save models/10x10/snake_10x10_100sess.keras

models-14:
	$(SNAKE) --sessions 1   --board-size 14 --training-mode single \
		--save models/14x14/snake_14x14_1sess.keras
	$(SNAKE) --sessions 10  --board-size 14 --training-mode single \
		--save models/14x14/snake_14x14_10sess.keras
	$(SNAKE) --sessions 100 --board-size 14 --training-mode single \
		--save models/14x14/snake_14x14_100sess.keras

models-18:
	$(SNAKE) --sessions 1   --board-size 18 --training-mode single \
		--save models/18x18/snake_18x18_1sess.keras
	$(SNAKE) --sessions 10  --board-size 18 --training-mode single \
		--save models/18x18/snake_18x18_10sess.keras
	$(SNAKE) --sessions 100 --board-size 18 --training-mode single \
		--save models/18x18/snake_18x18_100sess.keras

models-multi:
	$(SNAKE) --sessions 1   --training-mode multi \
		--save models/multi/snake_multi_1sess.keras
	$(SNAKE) --sessions 10  --training-mode multi \
		--save models/multi/snake_multi_10sess.keras
	$(SNAKE) --sessions 100 --training-mode multi \
		--save models/multi/snake_multi_100sess.keras

models: models-10 models-14 models-18

# ---------------------------------------------------------------------------
# Hyperparameter tuning targets (Optuna)
# Search space: learning_rate (log 1e-4..3e-3), gamma (0.92..0.99),
# batch_size {128,256,512,1024}, NO_EAT (-2..-0.1), 2-4 hidden layers
# of 32-256 units each.
# Pruning: lenient by default (percentile-25 + patience-2 + warmup 1500).
# Results: optuna_results/best_*.json + sqlite study in optuna_studies/.
# Resume: re-running 'make tune' appends new trials to the same study.
# ---------------------------------------------------------------------------

tune:
	$(VENV_PY) $(TUNE) --trials $(TRIALS) \
		--episodes-per-trial $(EPISODES_PER_TRIAL) \
		--training-mode single --board-size $(BOARD) \
		--pruner $(PRUNER) --prune-percentile $(PRUNE_PCT) \
		--patience $(PATIENCE) --n-warmup-steps $(N_WARMUP_STEPS)

tune-multi:
	$(VENV_PY) $(TUNE) --trials $(TRIALS) \
		--episodes-per-trial $(EPISODES_PER_TRIAL) \
		--training-mode multi \
		--pruner $(PRUNER) --prune-percentile $(PRUNE_PCT) \
		--patience $(PATIENCE) --n-warmup-steps $(N_WARMUP_STEPS)

tune-fast:
	$(VENV_PY) $(TUNE) --trials $(TRIALS) \
		--episodes-per-trial 500 \
		--training-mode single --board-size $(BOARD) \
		--pruner none

# Inspect an existing Optuna study: best params, top-N trials, parameter
# importance, and the median/range of high-value trials. Read-only.
tune-inspect:
	$(VENV_PY) src/AI_Model/inspect_study.py \
		--training-mode single --board-size $(BOARD)

tune-inspect-multi:
	$(VENV_PY) src/AI_Model/inspect_study.py \
		--training-mode multi
