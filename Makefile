PYTHON  ?= python3
VENV_PY := .venv/bin/python
PIP     := .venv/bin/pip
MAIN    := src/AI_Model/main.py

SESSIONS ?= 100
BOARD    ?= 10
FPS      ?= 10
# Default model resolves to the board-size subdirectory automatically.
# Override with MODEL=models/<size>x<size>/<file>.keras for a specific checkpoint.
MODEL    ?= models/$(BOARD)x$(BOARD)/best_snake_$(BOARD)x$(BOARD)_3000ep.keras
SAVE     ?=

.PHONY: all help create-venv install-deps setup \
	train train-visual train-debug train-continue \
	eval eval-visual eval-step \
	models models-10 models-14 models-18

all: setup

help:
	@echo "Available targets:"
	@echo "  make setup                                -> create venv and install dependencies"
	@echo ""
	@echo "  Training (models are saved to models/<BOARD>x<BOARD>/ automatically):"
	@echo "  make train         SESSIONS=500 BOARD=10 -> train without visualization"
	@echo "  make train-visual  SESSIONS=100 BOARD=14 -> train with pygame visualization"
	@echo "  make train-debug   SESSIONS=50  BOARD=10 -> train with debug output"
	@echo "  make train-continue              BOARD=10 -> resume training from MODEL"
	@echo ""
	@echo "  Evaluation (board size is read from the model metadata):"
	@echo "  make eval          BOARD=10 SESSIONS=30  -> evaluate without visualization"
	@echo "  make eval-visual   BOARD=14              -> evaluate with full UI"
	@echo "  make eval-step     BOARD=10              -> evaluate step-by-step"
	@echo ""
	@echo "  Model generation (required submission models):"
	@echo "  make models                              -> generate 1/10/100-session models for all board sizes"
	@echo "  make models-10                           -> generate models for 10x10 only"
	@echo "  make models-14                           -> generate models for 14x14 only"
	@echo "  make models-18                           -> generate models for 18x18 only"
	@echo ""
	@echo "  Variables (current values):"
	@echo "    SESSIONS=$(SESSIONS)  BOARD=$(BOARD)  FPS=$(FPS)"
	@echo "    MODEL=$(MODEL)"

create-venv:
	$(PYTHON) -m venv .venv
	@echo "Virtual environment created."

install-deps:
	$(PIP) install -r requirements.txt

setup: create-venv install-deps

# ---------------------------------------------------------------------------
# Training targets
# Models are saved automatically to models/<BOARD>x<BOARD>/ by main.py.
# Use SAVE=<path> to override the output path.
# ---------------------------------------------------------------------------

train:
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) --board-size $(BOARD) --visual off \
		$(if $(SAVE),--save $(SAVE),)

train-visual:
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) --board-size $(BOARD) --visual on \
		--fps $(FPS) $(if $(SAVE),--save $(SAVE),)

train-debug:
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) --board-size $(BOARD) --visual off \
		--debug $(if $(SAVE),--save $(SAVE),)

train-continue:
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) --board-size $(BOARD) --visual off \
		--load $(MODEL) $(if $(SAVE),--save $(SAVE),)

# ---------------------------------------------------------------------------
# Evaluation targets
# Board size is enforced from the model's metadata file; BOARD must match.
# MODEL defaults to models/<BOARD>x<BOARD>/best_snake_<BOARD>x<BOARD>_3000ep.keras.
# ---------------------------------------------------------------------------

eval:
	$(VENV_PY) $(MAIN) --dontlearn --load $(MODEL) --sessions $(SESSIONS) \
		--board-size $(BOARD) --visual off

eval-visual:
	$(VENV_PY) $(MAIN) --dontlearn --load $(MODEL) --sessions $(SESSIONS) \
		--board-size $(BOARD) --visual on --fps $(FPS) --step-by-step

eval-step:
	$(VENV_PY) $(MAIN) --dontlearn --load $(MODEL) --sessions $(SESSIONS) \
		--board-size $(BOARD) --visual off --step-by-step

# ---------------------------------------------------------------------------
# Model generation targets
# Creates the three reference models (1, 10, 100 sessions) for each board size.
# Models land in models/<size>x<size>/ as per main.py naming convention.
# Run 'make setup' first.
# ---------------------------------------------------------------------------

models-10:
	$(VENV_PY) $(MAIN) --sessions 1   --board-size 10 \
		--save models/10x10/snake_10x10_1sess.keras
	$(VENV_PY) $(MAIN) --sessions 10  --board-size 10 \
		--save models/10x10/snake_10x10_10sess.keras
	$(VENV_PY) $(MAIN) --sessions 100 --board-size 10 \
		--save models/10x10/snake_10x10_100sess.keras

models-14:
	$(VENV_PY) $(MAIN) --sessions 1   --board-size 14 \
		--save models/14x14/snake_14x14_1sess.keras
	$(VENV_PY) $(MAIN) --sessions 10  --board-size 14 \
		--save models/14x14/snake_14x14_10sess.keras
	$(VENV_PY) $(MAIN) --sessions 100 --board-size 14 \
		--save models/14x14/snake_14x14_100sess.keras

models-18:
	$(VENV_PY) $(MAIN) --sessions 1   --board-size 18 \
		--save models/18x18/snake_18x18_1sess.keras
	$(VENV_PY) $(MAIN) --sessions 10  --board-size 18 \
		--save models/18x18/snake_18x18_10sess.keras
	$(VENV_PY) $(MAIN) --sessions 100 --board-size 18 \
		--save models/18x18/snake_18x18_100sess.keras

models: models-10 models-14 models-18
