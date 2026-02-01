# Neuro-Symbolic Agent — Testing Instructions

## 1. Pre-flight check

- **Scripts**: `awaken.sh` and `run.sh` are executable (`chmod +x`).
- **Dashboard**: `src/dashboard.py` exists and imports `streamlit`.
- **Memory**: `src/memory/episodic.py` has the `get_recent(n)` method used by the UI.

## 2. Execute the awakening

1. Open a terminal in the project root (`/Users/mahmoudomar/Desktop/NSCA` or your repo root).
2. Run:
   ```bash
   ./awaken.sh
   ```
3. Wait for training to finish (Phase 5 encoder ~5 min, Phase 7 binder ~1–2 min). The script will then start the console Life Log (Phase 8). You can answer the prompt or press Enter to use the default, then exit when done.

## 3. Launch the Neuro-Link dashboard

From the same project root, run:

```bash
.venv/bin/streamlit run src/dashboard.py
```

## 4. Open the dashboard in your browser

- **URL**: **http://localhost:8501**
- Open this address in your browser. Streamlit will open it automatically if possible; otherwise paste the URL manually.

## 5. Test the recall system

1. **Generate memories**: In the left column (“The World”), click the **“Step (Generate Random Event)”** button **5 times**. Each click encodes one random Fashion-MNIST image and stores it in episodic memory. You should see the current image, uncertainty bar, and decision update, and the right column (“The Memory Palace”) fill with the last 10 memories (Step, Estimated Concept, Uncertainty).

2. **Test recall**: In the middle column (“The Chat”), in the text box **“Ask me to find something (e.g., Sneaker or 7)”**, type:
   ```text
   Sneaker
   ```
   and press Enter (or submit). The app will use the binder and `memory.recall()` to search for steps where the stored latent is similar to the “Sneaker” concept. You should see either:
   - **“I remember seeing a Sneaker at Step X, Step Y, …”** if any of the 5 observations were close to Sneakers, or  
   - **“I don’t recall seeing that.”** if none were above the similarity threshold.

3. (Optional) Click **“Clear memory”** in the right column to reset and repeat the test.
