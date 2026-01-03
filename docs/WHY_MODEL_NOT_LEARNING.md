# למה המודל לא לומד? (12.5% = Random Chance)

## הבעיה

אם ה-val_accuracy נשאר 12.5% ולא משתנה, זה אומר שהמודל **לא לומד בכלל** - הוא פשוט מנחש אקראית.

12.5% = 1/8 classes = Random chance

## סיבות אפשריות

### 1. **הנתונים לא נטענים נכון**

**איך לבדוק:**
```python
import pandas as pd
df = pd.read_csv('Data/Labels/dataset.csv')
print(f"Total samples: {len(df)}")
print(f"Expected: 226 samples")

# Check if files exist
from pathlib import Path
keypoints_dir = Path('Data/Keypoints/rawVideos')
for idx, row in df.head(10).iterrows():
    path = row['path']
    if path.startswith("keypoints/"):
        path = path.replace("keypoints/", "", 1)
    file_path = keypoints_dir / path
    exists = file_path.exists()
    print(f"{row['path']}: {'✅' if exists else '❌'}")
```

**פתרון:** ודא שכל הקבצים קיימים וניתנים לטעינה.

### 2. **Labels לא מאוזנים**

אם יש class אחד עם הרבה יותר samples, המודל ילמד לנחש את ה-class הזה.

**איך לבדוק:**
```python
df = pd.read_csv('Data/Labels/dataset.csv')
print(df.groupby('label').size())
```

**פתרון:** ודא שיש בערך אותו מספר samples לכל class.

### 3. **בעיה עם Normalization**

אם ה-normalization לא נכון, הנתונים יכולים להיות לא ניתנים ללמידה.

**איך לבדוק:**
```python
# After loading data
print(f"Mean: {np.mean(X_train):.4f}")
print(f"Std: {np.std(X_train):.4f}")
print(f"Min: {np.min(X_train):.4f}, Max: {np.max(X_train):.4f}")

# Should be approximately: Mean ~0, Std ~1
```

**פתרון:** אם זה לא נכון, נסה בלי normalization או עם normalization אחר.

### 4. **המודל לא מתאים**

אולי המודל גדול מדי או קטן מדי לנתונים.

**פתרון:** נסה מודל פשוט יותר:
```python
!python scripts/train_model.py --csv Data/Labels/dataset.csv \
  --keypoints-dir Data/Keypoints/rawVideos \
  --output-dir models \
  --batch-size 8 \
  --epochs 200 \
  --gru-units 64 \
  --num-gru-layers 1 \
  --dropout 0.2 \
  --learning-rate 0.001 \
  --patience 50
```

### 5. **Learning Rate לא מתאים**

אם learning rate נמוך מדי, המודל לא ילמד. אם גבוה מדי, המודל לא יתכנס.

**פתרון:** נסה learning rate שונה:
- נמוך: 0.0001
- בינוני: 0.001 (מומלץ)
- גבוה: 0.01

### 6. **הנתונים לא מספיק טובים**

אם ה-keypoints לא נכונים או לא מספיק שונים בין classes, המודל לא יכול ללמוד.

**איך לבדוק:**
```python
# Check if keypoints are different between classes
from scripts.data_loader import SignLanguageDataLoader
loader = SignLanguageDataLoader('Data/Labels/dataset.csv', 'Data/Keypoints/rawVideos')
splits = loader.get_all_splits()
X_train, y_train = splits['train']

# Check variance per class
for class_idx in range(8):
    class_data = X_train[y_train == class_idx]
    variance = np.var(class_data)
    print(f"Class {class_idx}: variance = {variance:.4f}")
```

## פתרון מהיר

נסה את זה - מודל פשוט יותר עם learning rate גבוה יותר:

```python
!python scripts/train_model.py --csv Data/Labels/dataset.csv \
  --keypoints-dir Data/Keypoints/rawVideos \
  --output-dir models \
  --batch-size 8 \
  --epochs 200 \
  --gru-units 64 \
  --num-gru-layers 1 \
  --dropout 0.1 \
  --learning-rate 0.01 \
  --patience 50
```

## איך לבדוק מה הבעיה

הרץ את זה ב-Colab:

```python
!python scripts/debug_model_training.py
```

זה יבדוק:
- האם הנתונים נטענים נכון
- האם ה-labels מאוזנים
- האם הנתונים שונים בין classes
- האם המודל יכול ללמוד בכלל

## אם כלום לא עובד

אז הבעיה היא בנתונים עצמם:
1. **ה-keypoints לא נכונים** - צריך לחלץ מחדש
2. **ה-keypoints לא מספיק שונים** - צריך יותר variation בנתונים
3. **ה-labels לא נכונים** - צריך לבדוק את ה-CSV

