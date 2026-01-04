# 🚀 איך לאמן מחדש את המודל בקולאב - מדריך מפורט

## 📋 שלב אחר שלב

### **שלב 1: לפתוח את הקולאב**

1. לפתוח את הקישור הזה:
   ```
   https://colab.research.google.com/github/MAya0M/SignLanguage-Recognition/blob/main/notebooks/SignLanguage_Training.ipynb
   ```
   
   או:
   - לפתוח את הקובץ `notebooks/SignLanguage_Training.ipynb` ב-GitHub
   - ללחוץ על הכפתור "Open in Colab" (התג הירוק)

### **שלב 2: להפעיל GPU** ⚠️ **חשוב מאוד!**

1. בתפריט העליון: **Runtime → Change runtime type**
2. ב-**Hardware accelerator**: לבחור **GPU**
3. ללחוץ **Save**

**למה זה חשוב?**
- אימון ב-GPU מהיר פי 10-50 מאשר CPU
- אימון ב-CPU יכול לקחת 4-6 שעות
- אימון ב-GPU לוקח 30-60 דקות

### **שלב 3: להעלות את הנתונים** 📁

**אם הנתונים כבר ב-GitHub:**
- לא צריך לעשות כלום! הקוד יוריד אותם אוטומטית

**אם הנתונים לא ב-GitHub:**
1. ליצור תיקייה `Data` בקולאב:
   ```python
   # להריץ בתא חדש
   !mkdir -p Data/Keypoints/rawVideos
   !mkdir -p Data/Labels
   ```

2. להעלות את הקבצים:
   - **אפשרות 1**: להעלות ידנית דרך Colab
     - ללחוץ על האייקון של התיקייה משמאל
     - ללחוץ על "Upload" ולהעלות את הקבצים
   
   - **אפשרות 2**: להעלות מ-Google Drive
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     # להעתיק את הקבצים מ-Drive
     ```

### **שלב 4: להריץ את כל התאים** ▶️

**אפשרות 1: להריץ הכל בבת אחת** (הכי קל)
1. בתפריט העליון: **Runtime → Run all**
2. לחכות שהכל יסתיים (30-60 דקות)

**אפשרות 2: להריץ תא אחר תא** (יותר שליטה)
1. ללחוץ על כל תא ולהריץ (Shift+Enter)
2. לחכות שכל תא יסתיים לפני התא הבא

### **שלב 5: לבדוק את התוצאות** 📊

לאחר שהאימון מסתיים, התא האחרון יראה:
- ✅ Test Accuracy
- ✅ Test Loss
- ✅ Best Validation Accuracy

**מה זה אומר?**
- **Accuracy > 80%**: מעולה! 🎉
- **Accuracy 60-80%**: טוב, אבל יכול להיות יותר טוב
- **Accuracy < 60%**: צריך לשפר (להוסיף נתונים, לשנות פרמטרים)

### **שלב 6: להוריד את המודל** 💾

1. להריץ את התא האחרון (Cell 12)
2. זה יוריד אוטומטית את המודל כקובץ ZIP
3. לחלץ את הקובץ
4. להעתיק את התיקייה `run_YYYYMMDD_HHMMSS` לתיקיית `models` בפרויקט המקומי

---

## ⚠️ בעיות נפוצות ופתרונות

### **בעיה 1: "No GPU available"**

**פתרון**:
1. Runtime → Change runtime type → GPU
2. אם עדיין לא עובד, לנסות:
   - Runtime → Disconnect and delete runtime
   - Runtime → Change runtime type → GPU
   - להריץ מחדש

### **בעיה 2: "Out of memory"**

**פתרון**:
1. להקטין את batch size:
   ```python
   --batch-size 4  # במקום 8
   ```
2. או להקטין את מספר ה-epochs:
   ```python
   --epochs 100  # במקום 200
   ```

### **בעיה 3: "Data not found"**

**פתרון**:
1. לבדוק שהנתונים בתיקייה הנכונה
2. להריץ:
   ```python
   !ls Data/
   !ls Data/Keypoints/rawVideos/
   !ls Data/Labels/
   ```
3. אם אין קבצים, להעלות אותם מחדש

### **בעיה 4: האימון לוקח יותר מדי זמן**

**פתרון**:
1. לוודא ש-GPU מופעל (Runtime → Change runtime type → GPU)
2. לבדוק:
   ```python
   import tensorflow as tf
   print("GPU Available:", tf.config.list_physical_devices('GPU'))
   ```
3. אם זה אומר "[]", GPU לא מופעל

---

## 📊 מה חדש באימון?

### **שיפורים שהוספנו:**

1. ✅ **Class Weights** - המודל עכשיו נותן יותר משקל למילים עם פחות סרטונים
   - זה עוזר למודל לא להעדיף את המילה הכי נפוצה (כמו HELLO)
   - אוטומטי - לא צריך לעשות כלום!

2. ✅ **Smart Frame Sampling** - המודל מתעלם מההתחלה הזהה ומתמקד במחווה
   - מדלג על 20% מההתחלה
   - מתמקד בחלק האמצעי/סופי (המחווה עצמה)
   - אוטומטי - לא צריך לעשות כלום!

3. ✅ **נורמליזציה מתוקנת** - אותה נורמליזציה באימון ובזיהוי
   - זה משפר את הדיוק

### **מה יראה באימון:**

```
Loading data...
⚠️  Using data WITHOUT additional normalization
✅ Using smart frame sampling (skips similar start, focuses on gesture)

Label distribution in train:
  HELLO: 45 samples
  YES: 32 samples
  ...

Class weights (to handle imbalance):
  HELLO: 0.850
  YES: 1.200
  ...

⚠️  Class imbalance detected (ratio: 2.50x)
   Using class weights to balance training
```

---

## 🎯 טיפים לשיפור התוצאות

### **אם ה-Accuracy נמוך (< 60%)**:

1. **להוסיף עוד סרטונים**:
   - לפחות 30-50 סרטונים לכל מילה
   - יותר מגוון: אנשים שונים, זוויות שונות

2. **לבדוק את Class Imbalance**:
   - התא הראשון (Cell 7) יראה את היחס בין המילים
   - אם יש חוסר איזון חמור (> 2.0x), להוסיף עוד סרטונים למילים הפחות נפוצות

3. **לשנות פרמטרים**:
   ```python
   --epochs 300  # יותר epochs
   --learning-rate 0.0005  # learning rate נמוך יותר
   --dropout 0.4  # יותר dropout
   ```

### **אם המודל עדיין מזהה רק מילה אחת**:

1. **לבדוק את הנתונים**:
   - האם יש מספיק סרטונים לכל מילה?
   - האם הסרטונים שונים מספיק?

2. **להוסיף עוד סרטונים**:
   - זה הפתרון הכי טוב!

3. **לבדוק את Class Weights**:
   - באימון, לבדוק שהמשקלות נראים הגיוניים
   - אם מילה אחת מקבלת משקל נמוך מדי, צריך להוסיף לה עוד סרטונים

---

## 📝 סיכום - צעדים מהירים

1. ✅ לפתוח את הקולאב
2. ✅ Runtime → Change runtime type → GPU
3. ✅ Runtime → Run all
4. ✅ לחכות 30-60 דקות
5. ✅ לבדוק את התוצאות
6. ✅ להוריד את המודל

**זה הכל!** 🚀

---

## ❓ שאלות נפוצות

**Q: כמה זמן לוקח האימון?**
A: 30-60 דקות עם GPU, 4-6 שעות עם CPU

**Q: מה אם הקולאב מתנתק?**
A: המודל נשמר אוטומטית! פשוט להריץ מחדש מהתא האחרון

**Q: איך יודעים שהאימון הסתיים?**
A: התא האחרון (Cell 10) יראה את התוצאות

**Q: מה אם ה-Accuracy נמוך?**
A: להוסיף עוד סרטונים ולאמן מחדש

**Q: איך להוריד את המודל?**
A: להריץ את Cell 12 - זה יוריד אוטומטית

---

**הכל מוכן!** פשוט לפתוח את הקולאב ולהריץ. 🎉

**שאלות?** תגיד לי אם משהו לא ברור!

