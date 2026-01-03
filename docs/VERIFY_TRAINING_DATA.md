# איך לוודא שהאימון משתמש בכל הסרטונים החדשים

## בעיה אפשרית

אם המודל לא לומד (accuracy נשאר ב-12.5% = random chance), ייתכן שה-CSV ב-Colab לא מעודכן.

## איך לבדוק

### 1. בדוק כמה סרטונים יש ב-CSV

בתא ב-Colab, הרץ:

```python
import pandas as pd
df = pd.read_csv('Data/Labels/dataset.csv')
print(f"Total samples: {len(df)}")
print(f"\nPer label:")
print(df.groupby('label').size())
print(f"\nPer split:")
print(df.groupby('split').size())
```

**צריך לראות:**
- **Total: 226 samples** (לא 129!)
- **28-29 samples per label** (לא 16!)
- **Train: 130, Val: 40, Test: 56**

### 2. בדוק אם יש קבצים 17-28

```python
# בדוק אם יש את הסרטונים החדשים
goodbye = df[df['label'] == 'GOODBYE']
files_17_28 = [p for p in goodbye['path'].tolist() 
               if any(str(i) in p for i in range(17, 29))]
print(f"GoodBye files 17-28: {len(files_17_28)} (should be 12)")
```

**צריך לראות:** 12 קבצים (17-28)

### 3. אם ה-CSV לא מעודכן

אם ה-CSV לא מכיל את כל 226 הסרטונים:

1. **ודא שה-CSV מעודכן ב-Git:**
   ```bash
   git pull
   cat Data/Labels/dataset.csv | wc -l  # צריך להיות 227 (226 + header)
   ```

2. **או העלה את ה-CSV מחדש:**
   - העלה את `Data/Labels/dataset.csv` ל-Google Drive
   - ב-Colab: `!cp /content/drive/MyDrive/dataset.csv Data/Labels/dataset.csv`

3. **או צור את ה-CSV מחדש ב-Colab:**
   ```python
   !python scripts/create_dataset_csv.py
   ```

### 4. בדוק שהקבצים קיימים

```python
from pathlib import Path
import numpy as np

keypoints_dir = Path('Data/Keypoints/rawVideos')
goodbye_dir = keypoints_dir / 'GoodBye'

# בדוק כמה קבצים יש
files = list(goodbye_dir.glob('*.npy'))
print(f"GoodBye keypoint files: {len(files)} (should be 28)")

# נסה לטעון קובץ
if files:
    test_file = files[0]
    data = np.load(test_file)
    print(f"✅ Loaded {test_file.name}: shape {data.shape}")
```

## אם הכל נכון אבל המודל עדיין לא לומד

אז הבעיה היא במודל עצמו, לא בנתונים. אפשר לנסות:

1. **בדוק את גודל ה-batch:**
   - עם 130 samples ב-train, batch size של 32 זה בסדר
   - אבל אם יש רק 16 samples, batch size של 32 זה גדול מדי

2. **בדוק את ה-learning rate:**
   - 0.001 זה בסדר, אבל אולי צריך לנסות 0.0001

3. **בדוק את הארכיטקטורה:**
   - אולי צריך יותר layers או יותר units

4. **בדוק את ה-preprocessing:**
   - אולי צריך normalization טוב יותר

