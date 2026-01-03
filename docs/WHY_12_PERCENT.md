# Why Model Stuck at 12.5% (Random Chance)

## The Problem
The model's accuracy is stuck at 12.5% (1/8 = random chance for 8 classes). This means the model cannot distinguish between different sign language words.

## Root Cause Analysis

### 1. Data Similarity
- **Average difference between classes: 0.046072** (very low!)
- Min difference: 0.016850
- Max difference: 0.090503
- **Classes are too similar after normalization**

### 2. Normalization Issues
The keypoints are normalized in `extract_keypoints.py`:
- Translate wrist to (0,0,0) - removes position differences
- Rotate hand to consistent direction - removes rotation differences  
- Scale by hand size - removes size differences

**This normalization removes important differences between classes!**

### 3. Small Dataset
- Only 16-17 samples per class
- Very small for deep learning (typically need 100+ per class)
- Model doesn't have enough examples to learn patterns

## Solutions (in order of effectiveness)

### ✅ BEST: Add More Training Data
- **Current:** 16-17 videos per word
- **Recommended:** 50+ videos per word
- **Why:** More examples = better learning
- **How:** Record more videos of each sign language word

### 🔧 Try Minimal Normalization
Instead of full normalization (translate + rotate + scale), try:
- **Only translate** wrist to origin (keep size/rotation differences)
- Or: **No normalization** at all
- **Why:** Preserves more differences between classes

### 🎯 Try Different Model Architecture
- **Current:** GRU (3 layers, 256 units)
- **Try:** 
  - LSTM (might learn better)
  - Transformer (attention mechanism)
  - CNN + RNN (spatial + temporal)
- **Why:** Different architectures learn different patterns

### 📈 Data Augmentation
- Add noise to keypoints
- Slight variations in timing
- Mirror/flip variations
- **Why:** Creates more training examples from existing data

### 🔍 Feature Engineering
Add more informative features:
- **Velocity:** How keypoints move (change over time)
- **Acceleration:** How movement changes
- **Relative positions:** Distances between keypoints
- **Why:** More features = more information for model to learn

## Immediate Action Plan

1. **Add more videos** (most important!)
   - Record 30-50 more videos per word
   - Ensure variety in hand position, size, speed
   
2. **Try minimal normalization**
   - Modify `extract_keypoints.py` to only translate (no rotate/scale)
   - Re-extract keypoints
   - Retrain model

3. **If still stuck:**
   - Try different model architecture
   - Add data augmentation
   - Add feature engineering

## Current Status
- ✅ Data loads correctly
- ✅ Labels are balanced
- ✅ Model can learn (loss decreases)
- ❌ Classes too similar (normalization removes differences)
- ❌ Too little data (16-17 samples per class)

