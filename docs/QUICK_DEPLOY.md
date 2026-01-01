# Quick Deploy - Get Your App Online in 5 Minutes! 🚀

## Step-by-Step Guide

### Option 1: Railway (Easiest - Recommended!) ⭐

1. **Go to Railway:**
   - Visit: https://railway.app
   - Click "Start a New Project"

2. **Sign in:**
   - Click "Login with GitHub"
   - Authorize Railway

3. **Deploy:**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Find and select: `MAya0M/SignLanguage-Recognition`
   - Railway will auto-detect it's a Flask app
   - Click "Deploy"

4. **Wait 2-3 minutes:**
   - Railway will build and deploy your app
   - You'll see a URL like: `https://your-app-name.railway.app`

5. **Done!** 🎉
   - Click the URL to see your app
   - Upload videos and get predictions!

**Note:** If you don't have a trained model yet, the app will show a warning. Train a model first (see below) or upload one.

---

### Option 2: Render (Also Easy!)

1. **Go to Render:**
   - Visit: https://render.com
   - Sign in with GitHub

2. **Create Web Service:**
   - Click "New" → "Web Service"
   - Select your repository: `MAya0M/SignLanguage-Recognition`

3. **Configure:**
   - **Name:** `sign-language-recognition` (or any name)
   - **Region:** Choose closest to you
   - **Branch:** `main` or `master`
   - **Root Directory:** (leave empty)
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`

4. **Deploy:**
   - Click "Create Web Service"
   - Wait 3-5 minutes
   - Your app will be at: `https://your-app-name.onrender.com`

5. **Done!** 🎉

---

### Option 3: Heroku

1. **Install Heroku CLI:**
   ```bash
   # Windows: Download from heroku.com
   # Mac: brew install heroku/brew/heroku
   # Linux: See heroku.com/install
   ```

2. **Login:**
   ```bash
   heroku login
   ```

3. **Create App:**
   ```bash
   heroku create your-app-name
   ```

4. **Deploy:**
   ```bash
   git push heroku main
   ```

5. **Open:**
   ```bash
   heroku open
   ```

---

## Before Deploying - Train a Model!

**Important:** The app needs a trained model to work!

### Quick Training (Google Colab):

1. Open: https://colab.research.google.com/github/MAya0M/SignLanguage-Recognition/blob/main/notebooks/SignLanguage_Training.ipynb
2. Runtime → Change runtime type → **GPU**
3. Runtime → Run all
4. Wait for training to complete
5. Download the model from `models/run_*/best_model.keras`
6. Upload to your deployed app (or include in repo)

### Or Train Locally:

```bash
python scripts/train_model.py --csv Data/Labels/dataset.csv
```

---

## After Deployment

### Your App URL:
- **Railway:** `https://your-app-name.railway.app`
- **Render:** `https://your-app-name.onrender.com`
- **Heroku:** `https://your-app-name.herokuapp.com`

### Test It:
1. Open the URL in your browser
2. Upload a sign language video
3. Click "Recognize Sign Language"
4. Get the prediction!

---

## Troubleshooting

### "No trained model found"
- Train a model first (see above)
- Or upload model to the deployed app

### "App not loading"
- Check deployment logs
- Make sure all dependencies are in `requirements.txt`
- Verify `Procfile` exists

### "Port error"
- Railway/Render/Heroku set PORT automatically
- Make sure `app.py` uses: `port = int(os.environ.get('PORT', 5000))`

---

## Auto-Deploy

Once connected to Railway/Render:
- **Every push to `main` branch = automatic redeploy!**
- No need to manually deploy again
- Just push code and it updates automatically

---

**That's it! Your app is now online! 🎉**

