# Railway Troubleshooting - Seeing Railway API Page Instead of App

## Problem

You see "✨ Home of the Railway API ✨" instead of your Flask app.

This means Railway is showing its default page because:
- The app isn't running/started properly
- The app crashed during startup
- The service isn't properly configured

## How to Fix

### Step 1: Check Logs

1. In Railway dashboard, click on your service
2. Click **"Logs"** tab (top navigation)
3. Look for errors or see if the app started

### Step 2: Check Deployment Status

1. Go to **"Deployments"** tab
2. Check if the latest deployment is **ACTIVE** (green)
3. If it shows errors (red), click on it to see details

### Step 3: Verify Requirements

Make sure you committed and pushed the fixes:
- ✅ `opencv-python-headless` (not `opencv-python`)
- ✅ `gunicorn>=21.2.0` added

If not, do:
```bash
git add requirements.txt
git commit -m "Fix Railway: opencv-headless and gunicorn"
git push
```

### Step 4: Check Service Configuration

1. Go to **"Settings"** tab
2. Check **"Start Command"** - should be empty (Procfile handles it)
3. Or set to: `gunicorn app:app`
4. Check **"Healthcheck Path"** - leave empty or set to `/`

### Step 5: Manual Redeploy

If still not working:
1. Go to **"Deployments"** tab
2. Click the three dots (⋯) on latest deployment
3. Click **"Redeploy"**
4. Wait for it to complete

## Common Issues

### Issue: App crashes on startup

**Check logs for:**
- Import errors
- Missing dependencies
- Port binding errors

**Solution:** Make sure all dependencies in `requirements.txt` and `app.py` uses `PORT` env variable (already fixed).

### Issue: Procfile not found

**Solution:** Make sure `Procfile` exists in root with:
```
web: gunicorn app:app
```

### Issue: Service not exposed

**Solution:** Make sure you generated a domain (exposed the service).

## What Logs Should Show (When Working)

When the app starts successfully, logs should show:
```
[INFO] Starting gunicorn...
[INFO] Listening at: http://0.0.0.0:PORT
[INFO] Application startup complete.
```

## Quick Checklist

- [ ] Committed and pushed latest `requirements.txt` changes
- [ ] Latest deployment shows ACTIVE (green)
- [ ] Service is exposed (has public URL)
- [ ] Logs show app started (not errors)
- [ ] `Procfile` exists with `web: gunicorn app:app`
- [ ] `app.py` uses `PORT` environment variable

## Still Not Working?

1. Check Railway logs carefully
2. Try redeploying manually
3. Verify all files are in GitHub repo
4. Check that `app.py` exists and is correct

