.PHONY: help eval submit all

# ==========================================
# TRAINING-FREE TRACK 
# ==========================================

# Run the local evaluation script to check the score
eval:
	@echo "==> Running local evaluation (Training-Free)..."
	uv run train-free/evaluate_local.py

# Generate the final test_predictions.csv
submit:
	@echo "==> Generating submission file (Training-Free)..."
	uv run train-free/submit_baseline.py

# Run both: check the score first, then build the submission
all: submit eval

# ==========================================
# MODEL TRAINING TRACK 
# ==========================================
# (Kamal's workspace - add commands like 'make train' or 'make cluster-sync' here later)


# ==========================================
# UTILITIES
# ==========================================

help:
	@echo "Available commands:"
	@echo "  make eval    - Run the local evaluation script (Training-Free)"
	@echo "  make submit  - Generate the test_predictions.csv (Training-Free)"
	@echo "  make all     - Run evaluation and then generate submission"