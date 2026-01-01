# Fix Gunicorn Error on Railway

## Problem

```
/bin/bash: line 1: gunicorn: command not found
```

This happens because `gunicorn` is not installed, but the `Procfile` tries to use it.

## Solution

Add `gunicorn` to `requirements.txt`.

**Already fixed!** Just commit and push.

## How to Fix

1. **The fix is already in requirements.txt** - `gunicorn>=21.2.0` added
2. **Commit and push:**
   ```bash
   git add requirements.txt
   git commit -m "Add gunicorn for Railway deployment"
   git push
   ```
3. **Railway will automatically redeploy**
4. **App should start successfully!**

## Why This is Needed

- Railway uses `Procfile` to know how to start the app
- `Procfile` contains: `web: gunicorn app:app`
- But `gunicorn` needs to be installed via `requirements.txt`

## Alternative (if gunicorn fails)

If you want to use Flask's built-in server (not recommended for production, but works):

Change `Procfile` to:
```
web: python app.py
```

And update `app.py` to use PORT environment variable (already done).

But **gunicorn is better** for production - it's what we added.

