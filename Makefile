PYTHON  ?= python3
VENV_PY := .venv/bin/python
PIP     := .venv/bin/pip
MAIN    := src/AI_Model/main.py

SESSIONS ?= 100
BOARD    ?= 10
FPS      ?= 10
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
	models models-10 models-14 models-18 models-multi

all: setup

help:
	@echo "Available targets:"
	@echo "  make setup                                -> create venv and install dependencies"
	@echo ""
	@echo "  Training — single mode (one model per board size):"
	@echo "  make train         SESSIONS=500 BOARD=10 -> train without visualization"
	@echo "  make train-visual  SESSIONS=100 BOARD=14 -> train with pygame visualization"
	@echo "  make train-debug   SESSIONS=50  BOARD=10 -> train with debug output"
	@echo "  make train-continue              BOARD=10 -> resume training from MODEL"
	@echo ""
	@echo "  Training — multi mode (one model for 10/14/18, bonus):"
	@echo "  make train-multi          SESSIONS=10000 -> multi-size training, weighted 70/15/15"
	@echo "  make train-multi-debug    SESSIONS=50    -> multi-size training with debug"
	@echo "  make train-multi-continue                -> resume multi-mode training from MODEL"
	@echo ""
	@echo "  Evaluation — single mode:"
	@echo "  make eval          BOARD=10 SESSIONS=30  -> evaluate without visualization"
	@echo "  make eval-visual   BOARD=14              -> evaluate with full UI"
	@echo "  make eval-step     BOARD=10              -> evaluate step-by-step in terminal"
	@echo ""
	@echo "  Evaluation — multi mode:"
	@echo "  make eval-multi          BOARD=10 SESSIONS=30  -> evaluate multi model on a specific size"
	@echo "  make eval-multi-visual                          -> evaluate with UI (3-size selector)"
	@echo ""
	@echo "  Model generation (required submission models):"
	@echo "  make models                              -> generate 1/10/100-session models for all sizes"
	@echo "  make models-10                           -> generate models for 10x10 only"
	@echo "  make models-14                           -> generate models for 14x14 only"
	@echo "  make models-18                           -> generate models for 18x18 only"
	@echo "  make models-multi                        -> generate 1/10/100-session multi models (bonus)"
	@echo ""
	@echo "  Variables (current values):"
	@echo "    SESSIONS=$(SESSIONS)  BOARD=$(BOARD)  FPS=$(FPS)  MODE=$(MODE)"
	@echo "    MODEL=$(MODEL)"

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
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) --board-size $(BOARD) \
		--training-mode single --visual off \
		$(if $(SAVE),--save $(SAVE),)

train-visual:
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) --board-size $(BOARD) \
		--training-mode single --visual on --fps $(FPS) \
		$(if $(SAVE),--save $(SAVE),)

train-debug:
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) --board-size $(BOARD) \
		--training-mode single --visual off --debug \
		$(if $(SAVE),--save $(SAVE),)

train-continue:
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) --board-size $(BOARD) \
		--training-mode single --visual off --load $(MODEL) \
		$(if $(SAVE),--save $(SAVE),)

# ---------------------------------------------------------------------------
# Training targets — multi mode (bonus: size-portable model)
# Each episode samples a board size from [10, 14, 18] with weights
# [0.7, 0.15, 0.15]. Models are saved under models/multi/.
# --board-size is irrelevant in this mode (overridden per episode).
# ---------------------------------------------------------------------------

train-multi:
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) \
		--training-mode multi --visual off \
		$(if $(SAVE),--save $(SAVE),)

train-multi-debug:
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) \
		--training-mode multi --visual off --debug \
		$(if $(SAVE),--save $(SAVE),)

train-multi-continue:
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) \
		--training-mode multi --visual off --load $(MODEL) \
		$(if $(SAVE),--save $(SAVE),)

# ---------------------------------------------------------------------------
# Evaluation targets — single mode
# Board size must match the model's training size; main.py reads it from
# the *_metadata.json and overrides --board-size if they disagree.
# ---------------------------------------------------------------------------

eval:
	$(VENV_PY) $(MAIN) --dontlearn --load $(MODEL) --sessions $(SESSIONS) \
		--board-size $(BOARD) --visual off

eval-visual:
	$(VENV_PY) $(MAIN) --dontlearn --load $(MODEL) --sessions $(SESSIONS) \
		--board-size $(BOARD) --visual on --fps $(FPS)

eval-step:
	$(VENV_PY) $(MAIN) --dontlearn --load $(MODEL) --sessions $(SESSIONS) \
		--board-size $(BOARD) --visual off --step-by-step

# ---------------------------------------------------------------------------
# Evaluation targets — multi mode
# eval-multi runs without UI on a specific BOARD (10/14/18).
# eval-multi-visual opens the UI lobby with the 3-size selector enabled.
# ---------------------------------------------------------------------------

eval-multi:
	$(VENV_PY) $(MAIN) --dontlearn --load $(MODEL) --sessions $(SESSIONS) \
		--board-size $(BOARD) --visual off

eval-multi-visual:
	$(VENV_PY) $(MAIN) --dontlearn --load $(MODEL) --sessions $(SESSIONS) \
		--visual on --fps $(FPS)

# ---------------------------------------------------------------------------
# Model generation targets
# Creates reference models for each board size.
# Run 'make setup' first.
# ---------------------------------------------------------------------------

models-10:
	$(VENV_PY) $(MAIN) --sessions 1   --board-size 10 --training-mode single \
		--save models/10x10/snake_10x10_1sess.keras
	$(VENV_PY) $(MAIN) --sessions 10  --board-size 10 --training-mode single \
		--save models/10x10/snake_10x10_10sess.keras
	$(VENV_PY) $(MAIN) --sessions 100 --board-size 10 --training-mode single \
		--save models/10x10/snake_10x10_100sess.keras

models-14:
	$(VENV_PY) $(MAIN) --sessions 1   --board-size 14 --training-mode single \
		--save models/14x14/snake_14x14_1sess.keras
	$(VENV_PY) $(MAIN) --sessions 10  --board-size 14 --training-mode single \
		--save models/14x14/snake_14x14_10sess.keras
	$(VENV_PY) $(MAIN) --sessions 100 --board-size 14 --training-mode single \
		--save models/14x14/snake_14x14_100sess.keras

models-18:
	$(VENV_PY) $(MAIN) --sessions 1   --board-size 18 --training-mode single \
		--save models/18x18/snake_18x18_1sess.keras
	$(VENV_PY) $(MAIN) --sessions 10  --board-size 18 --training-mode single \
		--save models/18x18/snake_18x18_10sess.keras
	$(VENV_PY) $(MAIN) --sessions 100 --board-size 18 --training-mode single \
		--save models/18x18/snake_18x18_100sess.keras

models-multi:
	$(VENV_PY) $(MAIN) --sessions 1   --training-mode multi \
		--save models/multi/snake_multi_1sess.keras
	$(VENV_PY) $(MAIN) --sessions 10  --training-mode multi \
		--save models/multi/snake_multi_10sess.keras
	$(VENV_PY) $(MAIN) --sessions 100 --training-mode multi \
		--save models/multi/snake_multi_100sess.keras

models: models-10 models-14 models-18
