# השוואת AMIs - איזה לבחור?

## Deep Learning Base AMI vs Deep Learning AMI

### מה שמצאת: **Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)**

**יתרונות:**
- ✅ Ubuntu 22.04 (חדש)
- ✅ CUDA מותקן
- ✅ GPU drivers מוכנים
- ✅ יותר קל משקל

**חסרונות:**
- ❌ פחות כלים מותקנים מראש
- ❌ צריך להתקין יותר דברים ידנית (TensorFlow, Python packages, וכו')
- ❌ יותר עבודה בהכנה

---

### מומלץ יותר: **Deep Learning AMI (Ubuntu)**

**יתרונות:**
- ✅ TensorFlow מותקן מראש
- ✅ PyTorch מותקן מראש
- ✅ Python packages נפוצים
- ✅ פחות עבודה בהכנה
- ✅ הכל מוכן לאימון

**חסרונות:**
- ⚠️ קצת יותר גדול
- ⚠️ לוקח יותר זמן להוריד

---

## המלצה לפרויקט שלך

### אפשרות 1: Deep Learning AMI (מומלץ!)

**למה?**
- TensorFlow כבר מותקן
- פחות עבודה
- יותר מהיר להתחיל

**איך למצוא:**
1. ב-Launch Instance
2. בחיפוש AMI: `Deep Learning AMI (Ubuntu)`
3. בחר את הגרסה האחרונה

### אפשרות 2: Deep Learning Base AMI (יכול לעבוד)

**למה?**
- יותר קל משקל
- יותר שליטה

**מה צריך לעשות אחר כך:**
- להתקין TensorFlow: `pip install tensorflow`
- להתקין תלויות אחרות
- יותר עבודה

---

## מה לעשות עם Base AMI?

אם כבר בחרת את Base AMI, זה בסדר! תצטרך:

```bash
# ב-EC2 אחרי התחברות
sudo apt-get update
sudo apt-get install python3-pip -y
pip3 install tensorflow numpy pandas scikit-learn opencv-python mediapipe boto3
```

**זה יעבוד, אבל יותר עבודה.**

---

## המלצה סופית

**אם יש לך זמן:**
- חפש "Deep Learning AMI (Ubuntu)" - יותר נוח

**אם כבר בחרת Base AMI:**
- זה בסדר! תצטרך רק להתקין עוד כמה דברים

**שניהם יעבדו!** זה רק שאלה של נוחות.

---

## איך לשנות AMI?

אם עוד לא Launchת את ה-instance:
1. חזור ל-Launch Instance
2. חפש "Deep Learning AMI (Ubuntu)"
3. בחר אותו

אם כבר Launchת:
- זה בסדר, Base AMI יעבוד גם!

---

**סיכום: Base AMI יעבוד, אבל Deep Learning AMI המלא יותר נוח!**

