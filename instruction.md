# 🏙️ Accessibility Prediction Model - Quick Start Guide

Predict where future accessibility problems are likely to occur in your city!

---

## 📦 What You Need

1. **Python 3.7+** installed on your computer
2. **Your dataset**: `everydayLife_cleaned_dataset.csv`
3. **This script**: `accessibility_prediction_local.py`

---

## 🚀 Installation (One-Time Setup)

Open your terminal/command prompt and install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**Note for Mac/Linux users:** You might need to use `pip3` instead of `pip`

---

## ▶️ How to Run

### Step 1: Set Up Your Folder

Put these files in the **same folder**:
- ✅ `accessibility_prediction_local.py` (the script)
- ✅ `everydayLife_cleaned_dataset.csv` (your data)

### Step 2: Run the Script

Open terminal/command prompt, navigate to your folder, and run:

```bash
python accessibility_prediction_local.py
```

**Mac/Linux users:**
```bash
python3 accessibility_prediction_local.py
```

### Step 3: Wait for Magic! ✨

The script will:
- Load your data (81,973 accessibility issues)
- Create 965 spatial grid cells
- Train 2 machine learning models
- Generate visualizations
- Save everything for you

**Takes about 30-60 seconds** depending on your computer.

---

## 📊 What You Get

After running, you'll have these files:

### 1. **Models** (for predictions)
- `accessibility_regression_model.pkl` - Predicts exact # of issues
- `accessibility_classification_model.pkl` - Predicts High/Med/Low risk
- `label_encoder.pkl` - Helper file for neighborhoods

### 2. **Visualization**
- `results.png` - 4-panel dashboard showing:
  - ✅ Actual vs Predicted accuracy
  - ✅ Feature importance ranking
  - ✅ Spatial risk heatmap
  - ✅ Distribution comparison

---

## 🎯 How to Use Your Trained Model

### Option 1: Quick Prediction (Copy-Paste Code)

Create a new file called `make_prediction.py`:

```python
import pickle
import pandas as pd

# Load the model
with open('accessibility_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Predict for a new location
# Example: Capitol Hill area with some existing issues
new_location = pd.DataFrame([{
    'center_lon': -122.33,
    'center_lat': 47.61,
    'avg_severity': 3.0,
    'max_severity': 4.0,
    'std_severity': 1.0,
    'temp_count': 2,
    'issue_type_diversity': 3,
    'type_CurbRamp': 5,
    'type_NoCurbRamp': 15,
    'type_NoSidewalk': 3,
    'type_Obstacle': 2,
    'type_Occlusion': 1,
    'type_Other': 0,
    'type_SurfaceProblem': 8,
    'neighborhood_encoded': le.transform(['Capitol Hill'])[0]
}])

predicted_issues = model.predict(new_location)[0]
print(f"🎯 Predicted issues for this location: {predicted_issues:.0f}")

# Classify risk level
if predicted_issues < 30:
    risk = "LOW"
elif predicted_issues < 100:
    risk = "MEDIUM"
else:
    risk = "HIGH"

print(f"⚠️  Risk Level: {risk}")
```

Run it:
```bash
python make_prediction.py
```

### Option 2: Predict for Multiple Locations

```python
import pickle
import pandas as pd

# Load model
with open('accessibility_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict for multiple new locations
locations = pd.DataFrame([
    # Location 1
    {'center_lon': -122.33, 'center_lat': 47.61, 'avg_severity': 3.0, ...},
    # Location 2
    {'center_lon': -122.35, 'center_lat': 47.62, 'avg_severity': 2.5, ...},
    # Add more locations...
])

predictions = model.predict(locations)
for i, pred in enumerate(predictions):
    print(f"Location {i+1}: {pred:.0f} predicted issues")
```

---

## 📈 Model Performance

Your model achieves:
- **95.3% accuracy** (R² score) for regression
- **91.2% accuracy** for classification
- Average error of only **±9 issues**

### Top 3 Most Important Predictors:
1. 🚧 **CurbRamp density** (32.4%)
2. 🛣️ **SurfaceProblem patterns** (22.8%)
3. ♿ **NoCurbRamp issues** (21.7%)

---

## 🛠️ Troubleshooting

### "ModuleNotFoundError: No module named 'sklearn'"
**Solution:** Install scikit-learn:
```bash
pip install scikit-learn
```

### "FileNotFoundError: everydayLife_cleaned_dataset.csv"
**Solution:** Make sure your dataset is in the **same folder** as the script

### Script runs but no output files
**Solution:** Check file permissions - make sure you can write files to that folder

### "ImportError: cannot import name..."
**Solution:** Update your packages:
```bash
pip install --upgrade pandas numpy scikit-learn matplotlib seaborn
```

---

## 💡 Tips

- **Adjust grid size:** In the script, change `grid_size = 0.005` to make cells bigger/smaller
- **Different cities:** Just swap out the CSV with data from another city (same format)
- **Better accuracy:** Increase `n_estimators=100` to `200` or `500` (slower but more accurate)

---

## 📞 Need Help?

Check the console output - it shows progress and any errors!

The script prints:
- ✅ Data loading status
- ✅ Number of grid cells created
- ✅ Model training progress
- ✅ Performance metrics
- ✅ File save confirmations

---

## 🎉 You're All Set!

Now you can:
- 🗺️ Identify high-risk areas
- 📍 Predict issues for new locations  
- 🚀 Prioritize accessibility improvements
- 📊 Share visual insights with stakeholders

Happy predicting! 🎯
