# מדריך Deployment - פרסום האפליקציה לאינטרנט

אחרי שה-GitHub Actions workflow מסיים בהצלחה, אפשר לפרסם את האפליקציה לאינטרנט!

---

## אפשרויות Deployment (מומלץ)

### 1. **Railway** ⭐ (מומלץ ביותר!)

**יתרונות:**
- ✅ **חינם** - $5 credit חינם כל חודש
- ✅ **קל** - חיבור ישיר ל-GitHub
- ✅ **אוטומטי** - deploy בכל push
- ✅ **URL קבוע**

**איך לעשות:**

1. לך ל-https://railway.app
2. התחבר עם GitHub
3. **New Project** → **Deploy from GitHub repo**
4. בחר את ה-repository שלך
5. Railway יזהה אוטומטית שזה Flask app
6. **Deploy** - זה הכל!

**האפליקציה תהיה זמינה ב-URL כזה:**
```
https://your-app-name.railway.app
```

---

### 2. **Render**

**יתרונות:**
- ✅ **חינם** - עם limitations
- ✅ **קל**
- ✅ **אוטומטי**

**איך לעשות:**

1. לך ל-https://render.com
2. התחבר עם GitHub
3. **New** → **Web Service**
4. בחר repository
5. הגדרות:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
6. **Create Web Service**

---

### 3. **Heroku**

**יתרונות:**
- ✅ חינם (עם limitations)
- ✅ פופולרי

**דרישות:**
- `Procfile` - קובץ שמגדיר איך להריץ את האפליקציה

צור `Procfile`:
```
web: gunicorn app:app
```

ואז deploy דרך Heroku CLI או GitHub integration.

---

### 4. **Fly.io**

**יתרונות:**
- ✅ **חינם** - 3 VMs חינם
- ✅ מהיר
- ✅ גלובלי

---

## איך לגרום ל-Deployment להיות אוטומטי?

### אפשרות 1: Railway Auto-Deploy

1. ב-Railway, בחיבור ל-GitHub
2. בחר **"Auto-Deploy"**
3. כל push ל-`main` = deploy אוטומטי!

### אפשרות 2: GitHub Actions + Platform API

ניתן להוסיף ל-`.github/workflows/deploy.yml` deployment אוטומטי, אבל זה דורש:
- API keys
- הגדרות נוספות

**המומלץ:** להשתמש ב-auto-deploy של הפלטפורמה (Railway/Render).

---

## דרישות למודל

**חשוב:** האפליקציה צריכה את המודל המאומן!

**אפשרויות:**

### 1. העלה מודל ל-GitHub Releases
- צור release עם המודל
- הורד ב-deployment

### 2. העלה ל-Google Drive / Dropbox
- הורד ב-deployment

### 3. השאר ב-`models/` folder
- אם הקבצים קטנים, אפשר לשאול ב-Git
- ⚠️ לא מומלץ אם המודל גדול

---

## הגדרות נוספות

### Environment Variables

אם צריך:
```bash
FLASK_ENV=production
MODEL_DIR=models
```

ב-Railway/Render: Settings → Environment Variables

### Static Files

אם יש static files, תוסיף ל-`app.py`:
```python
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)
```

---

## Checklist לפני Deployment

- [ ] האפליקציה עובדת מקומית (`python app.py`)
- [ ] יש מודל מאומן ב-`models/run_*/best_model.keras`
- [ ] כל התלויות ב-`requirements.txt`
- [ ] `app.py` עובד
- [ ] `templates/index.html` קיים

---

## Troubleshooting

### "No model found"

- ודא שיש מודל ב-`models/run_*/best_model.keras`
- או העלה מודל דרך Google Drive

### "Module not found"

- ודא ש-`requirements.txt` מלא
- ב-Railway/Render: בדוק Build Logs

### "Port already in use"

- ב-Production משתמשים ב-Gunicorn:
  ```bash
  gunicorn app:app
  ```

---

**בהצלחה! 🚀**

