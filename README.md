# 🏏 IPL Match Winner Predictor: RCB vs PBKS (AI-based)

This project predicts the outcome of **RCB vs PBKS** matches using past IPL player performance data and a neural network model built with TensorFlow and Keras.

## 📌 What It Does

- Trains a neural network on **synthetically generated matchups** using real IPL player statistics.
- Calculates detailed team-level features like batting strength, bowling economy, all-rounder count, and player experience.
- Uses those features to predict **winning probabilities** between two teams — especially **RCB** and **PBKS**.
- Supports both **automated demo mode** and an **interactive team builder**.

---

## 🧠 How It Works

### ➤ Key Components:

- **Data Cleaning**: Handles missing/infinite values, caps outliers, and normalizes values.
- **Feature Engineering**: Computes metrics for each team based on the selected players (batting average, economy, depth, experience, etc.).
- **Synthetic Training Data**: Simulates 3000+ realistic IPL matches by randomly assembling balanced teams and computing their chances of winning.
- **Neural Network Model**: 
  - Deep model with dropout and batch normalization
  - Optimized with early stopping and learning rate scheduling
- **Prediction Engine**:
  - Accepts any 2 team lists (RCB, PBKS, or custom)
  - Outputs: predicted winner, win probabilities, key comparative metrics

---

## 🧪 Example Usage

```python
# Predict RCB vs PBKS
predictor = CorrectedIPLNeuralPredictor(data_path="IPL-Player-Stat.csv")

rcb_team = ['F du Plessis', 'V Kohli', 'GJ Maxwell', ...]
pbks_team = ['S Dhawan', 'LS Livingstone', 'SM Curran', ...]

result = predictor.predict_match_winner(rcb_team, pbks_team)

print(result["winner"])             # 'Team 1' or 'Team 2'
print(result["team1_win_prob"])     # RCB probability %
print(result["team2_win_prob"])     # PBKS probability %
