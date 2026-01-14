# Adaptive Split SplitFed (PyTorch)

Implements:
- Fixed split baseline
- Adaptive split per-client per-round
- Optional prototype-conditioned personalization

## Install
pip install -r requirements.txt

## Run
python run.py --mode fixed --no-proto
python run.py --mode adaptive
