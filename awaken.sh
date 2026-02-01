#!/bin/bash

# 1. Activate the Virtual Environment
source .venv/bin/activate

# 2. Train the Vision System (The "Eyes")
# This teaches it what a "Shoe" looks like vs a "Shirt"
echo "👁️  Training Vision System (Phase 5)..."
python src/world_model/train_encoder.py

# 3. Train the Language System (The "Ears")
# This teaches it that the word "Shoe" matches the visual concept of a Shoe
echo "👂 Training Language Grounding (Phase 7)..."
python src/language/train_grounding.py

# 4. Run the Full Life Simulation (Phase 8)
# It watches a movie, remembers it, and answers your questions.
echo "🧠 Starting Life Simulation (Phase 8)..."
python src/main.py
