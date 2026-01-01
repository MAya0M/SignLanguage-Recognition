# אוטומציה מלאה עם Google Colab

## פתרון אוטומטי - הכפתור "Open in Colab"

כשאתה עושה push ל-GitHub, הפרויקט כבר מוכן לעבודה ב-Colab!

### איך זה עובד:

1. **GitHub מכיר ב-`.ipynb` files** - כל קובץ notebook ב-GitHub מקבל כפתור "Open in Colab" אוטומטית
2. **הכפתור בפני עצמו** - ה-README כולל כפתור "Open in Colab" שמוביל ישירות ל-notebook
3. **הכל מוכן** - ה-notebook כבר מוגדר עם כל השלבים

---

## שימוש

### דרך 1: דרך GitHub

1. לך ל-https://github.com/MAya0M/SignLanguage-Recognition
2. לחץ על `notebooks/SignLanguage_Training.ipynb`
3. לחץ על **"Open in Colab"** (כפתור בחלק העליון)
4. **Runtime → Change runtime type → GPU**
5. **Runtime → Run all**

### דרך 2: דרך README

1. לך ל-https://github.com/MAya0M/SignLanguage-Recognition
2. לחץ על הכפתור **"Open in Colab"** ב-README (בחלק העליון)
3. **Runtime → Change runtime type → GPU**
4. **Runtime → Run all**

### דרך 3: ישירות ב-Colab

1. לך ל-https://colab.research.google.com
2. **File → Open notebook → GitHub**
3. הזן: `MAya0M/SignLanguage-Recognition`
4. בחר: `notebooks/SignLanguage_Training.ipynb`
5. **Runtime → Change runtime type → GPU**
6. **Runtime → Run all**

---

## מה ה-Notebook עושה אוטומטית:

1. ✅ **Clone** את ה-repository מ-GitHub
2. ✅ **בודק GPU** availability
3. ✅ **מתקין** את כל התלויות
4. ✅ **בודק** שכל הנתונים קיימים
5. ✅ **מאמן** את המודל
6. ✅ **שומר** את התוצאות

**זה הכל! אין צורך בהגדרה ידנית.**

---

## העלאת נתונים

אם צריך להעלות נתונים, ראה [COLAB_UPLOAD_GUIDE.md](COLAB_UPLOAD_GUIDE.md)

---

## טיפים

- **שמור את ה-notebook ב-Drive** - File → Save a copy in Drive
- **השתמש ב-GPU** - Runtime → Change runtime type → GPU (T4)
- **Run all** - Runtime → Run all (יותר נוח מ-Run cell אחד אחד)

---

**הכל אוטומטי! רק צריך ללחוץ על כפתור ולהריץ! 🚀**

