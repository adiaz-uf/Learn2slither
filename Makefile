PYTHON ?= python3
VENV_PY := .venv/bin/python
PIP := .venv/bin/pip
MAIN := src/AI_Model/main.py

SESSIONS ?= 100
BOARD ?= 10
FPS ?= 10
MODEL ?= models/best_snake_10x10_300ep.keras
SAVE ?=

.PHONY: all help create-venv install-deps setup \
	train train-visual train-debug train-continue \
	eval eval-visual eval-step

all: setup

help:
	@echo "Targets disponibles:"
	@echo "  make setup                           -> crear venv + instalar deps"
	@echo "  make train SESSIONS=500 BOARD=10     -> entrenar sin visual"
	@echo "  make train-visual SESSIONS=100 FPS=8 -> entrenar con visual"
	@echo "  make train-debug SESSIONS=50         -> entrenar con debug"
	@echo "  make train-continue MODEL=...        -> continuar entrenamiento"
	@echo "  make eval MODEL=... SESSIONS=30      -> evaluar sin aprendizaje"
	@echo "  make eval-visual MODEL=...           -> evaluar con UI"
	@echo "  make eval-step MODEL=...             -> evaluar paso a paso"
	@echo ""
	@echo "Variables:"
	@echo "  SESSIONS=$(SESSIONS) BOARD=$(BOARD) FPS=$(FPS)"
	@echo "  MODEL=$(MODEL)"

create-venv:
	$(PYTHON) -m venv .venv
	@echo "Virtual environment created"

install-deps:
	$(PIP) install -r requirements.txt

setup: create-venv install-deps

train:
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) --board-size $(BOARD) --visual off $(if $(SAVE),--save $(SAVE),)

train-visual:
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) --board-size $(BOARD) --visual on --fps $(FPS) $(if $(SAVE),--save $(SAVE),)

train-debug:
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) --board-size $(BOARD) --visual off --debug $(if $(SAVE),--save $(SAVE),)

train-continue:
	$(VENV_PY) $(MAIN) --sessions $(SESSIONS) --board-size $(BOARD) --visual off --load $(MODEL) $(if $(SAVE),--save $(SAVE),)

eval:
	$(VENV_PY) $(MAIN) --dontlearn --load $(MODEL) --sessions $(SESSIONS) --board-size $(BOARD) --visual off

eval-visual:
	$(VENV_PY) $(MAIN) --dontlearn --load $(MODEL) --sessions $(SESSIONS) --board-size $(BOARD) --visual on --fps $(FPS)

eval-step:
	$(VENV_PY) $(MAIN) --dontlearn --load $(MODEL) --sessions $(SESSIONS) --board-size $(BOARD) --visual off --step-by-step
