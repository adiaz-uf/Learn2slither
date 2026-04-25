# Learn2Slither - Usage Guide

A DQN-based Snake agent that learns through reinforcement learning.

---

## Setup

```bash
make setup          # create .venv and install dependencies
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Project structure

```
Learn2slither/
├── src/
│   ├── AI_Model/
│   │   ├── main.py       # entry point
│   │   ├── Agent.py      # DQN agent (experience replay + target network)
│   │   ├── Board.py      # game logic, rewards, vision
│   │   └── Snake.py      # snake data structure
│   └── UI/
│       └── game_ui.py    # pygame screens (board, game-over, stats)
├── models/
│   ├── 10x10/            # models trained on 10x10 board
│   ├── 14x14/            # models trained on 14x14 board
│   └── 18x18/            # models trained on 18x18 board
├── Makefile
└── requirements.txt
```

---

## Board sizes

Three board sizes are supported: **10**, **14**, **18** (playable cells; walls are added internally).

A model trained for a given board size **can only be used with that same board size**.
If the sizes do not match, the program prints an error and exits without starting.

Models are saved automatically under `models/<size>x<size>/`.

---

## Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--sessions N` | `100` | Number of training episodes |
| `--board-size {10,14,18}` | `10` | Board size |
| `--load PATH` | — | Load a pre-trained model |
| `--save PATH` | auto | Override the output file path |
| `--dontlearn` | off | Evaluation mode: run without updating weights |
| `--visual {on,off}` | `off` | Enable pygame visualization |
| `--step-by-step` | off | Pause after every step (evaluation only) |
| `--fps N` | `10` | Rendering speed (only with `--visual on`) |
| `--debug` | off | Detailed output for the first two episodes |

---

## Training

Models are saved to `models/<size>x<size>/` automatically.

```bash
# Train 500 episodes on a 10x10 board (no visualization, fastest)
python src/AI_Model/main.py --sessions 500 --board-size 10

# Train on a 14x14 board with visualization
python src/AI_Model/main.py --sessions 500 --board-size 14 --visual on --fps 10

# Resume training from an existing model
python src/AI_Model/main.py --sessions 500 --board-size 10 \
  --load models/10x10/snake_10x10_500ep_20260310_120000.keras

# Train with debug output (detailed for first 2 episodes)
python src/AI_Model/main.py --sessions 50 --board-size 10 --debug
```

Using make:

```bash
make train          BOARD=10 SESSIONS=500
make train          BOARD=14 SESSIONS=500
make train-visual   BOARD=10 SESSIONS=100 FPS=10
make train-debug    BOARD=10 SESSIONS=50
make train-continue BOARD=10 SESSIONS=500   # resumes from MODEL
```

Each run produces two files:

```
models/10x10/snake_10x10_500ep_20260310_120000.keras
models/10x10/snake_10x10_500ep_20260310_120000_metadata.json
```

The metadata stores `board_size`, `total_episodes`, `high_score`, `average_score`,
`final_epsilon`, and `timestamp`. It is read automatically when loading a model.

---

## Evaluation

The agent plays with no weight updates (pure exploitation, epsilon = 0).
Vision and action are printed to the terminal at every step.

```bash
# Evaluate 10 games in terminal mode
python src/AI_Model/main.py --dontlearn --board-size 10 \
  --load models/10x10/snake_10x10_500ep_20260310_120000.keras

# Evaluate with pygame visualization
python src/AI_Model/main.py --dontlearn --board-size 10 --visual on \
  --load models/10x10/snake_10x10_500ep_20260310_120000.keras

# Step-by-step: press Enter (terminal) or any key (visual) to advance
python src/AI_Model/main.py --dontlearn --board-size 10 --step-by-step \
  --load models/10x10/snake_10x10_500ep_20260310_120000.keras
```

Using make (MODEL defaults to `models/<BOARD>x<BOARD>/best_snake_<BOARD>x<BOARD>_3000ep.keras`):

```bash
make eval           BOARD=10 SESSIONS=10
make eval-visual    BOARD=10
make eval-step      BOARD=10
make eval           BOARD=14 MODEL=models/14x14/snake_14x14_100sess.keras SESSIONS=5
```

When `--visual on` is combined with `--dontlearn`, the full UI runs: after each game
a game-over screen appears with options to start a new game or view cumulative session
statistics (total games, total score, high score, average score).

---

## Generating reference models

The subject requires at least three models trained with 1, 10, and 100 sessions:

```bash
make models-10      # generates 1/10/100-session models for 10x10
make models-14      # generates 1/10/100-session models for 14x14
make models-18      # generates 1/10/100-session models for 18x18
make models         # all three sizes at once
```

---

## Expected performance (10x10 board)

| Episodes | Avg score | High score |
|---|---|---|
| 0-50 | 2-4 | 4-6 |
| 50-200 | 4-8 | 10-15 |
| 200-500 | 8-15 | 20-30 |
| 500-1000 | 15-25 | 35-50 |
| 1000+ | 20-30 | 50+ |

Training without visualization is roughly 10x faster than with `--visual on`.

---

## Troubleshooting

**Board-size mismatch when loading a model**
The program reads the board size from the model's metadata and exits if it does not
match `--board-size`. Use the same size the model was trained with.

**Score stuck at 2-3 after many episodes**
Run with `--debug` to inspect Q-values and loss, then consider training longer.

**`ModuleNotFoundError`**
Run `pip install -r requirements.txt` (or `make setup`) from the project root.

**Must run from the project root**
```bash
cd Learn2slither
python src/AI_Model/main.py --sessions 100
```
