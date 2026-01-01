# איך לבדוק את הנתונים ב-Colab

## בדיקת נתונים - הוסף ב-Colab

```python
# בדוק שהנתונים נטענים נכון
import numpy as np
import pandas as pd

# טען CSV
df = pd.read_csv('Data/Labels/dataset.csv')
print("CSV loaded:", len(df), "samples")
print(df.head())

# טען דוגמה אחת
from scripts.data_loader import SignLanguageDataLoader

loader = SignLanguageDataLoader('Data/Labels/dataset.csv', 'Data/Keypoints/rawVideos')

# טען דוגמה אחת
sample_path = df.iloc[0]['path']
print(f"\nLoading sample: {sample_path}")

keypoints = loader.load_keypoints(sample_path)
print(f"Keypoints shape: {keypoints.shape}")
print(f"Keypoints min: {keypoints.min()}, max: {keypoints.max()}")
print(f"Keypoints mean: {keypoints.mean()}, std: {keypoints.std()}")

# בדוק אם יש NaN או inf
print(f"Has NaN: {np.isnan(keypoints).any()}")
print(f"Has Inf: {np.isinf(keypoints).any()}")
```

## אם יש NaN או Inf:
- הבעיה בנרמול!
- צריך לתקן את normalize_keypoints

## אם הנתונים תקינים:
- אולי צריך יותר נתונים
- אולי המודל לא מתאים

