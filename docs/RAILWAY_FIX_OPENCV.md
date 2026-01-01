# Fix OpenCV Error on Railway

## Problem

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

This happens because `opencv-python` requires GUI libraries (libGL) that don't exist in Railway's headless (no display) environment.

## Solution

Use `opencv-python-headless` instead of `opencv-python`.

**Already fixed in requirements.txt!** Just redeploy.

## How to Fix

1. **The fix is already in requirements.txt** - changed to `opencv-python-headless`
2. **Commit and push to GitHub:**
   ```bash
   git add requirements.txt
   git commit -m "Fix opencv for Railway: use headless version"
   git push
   ```
3. **Railway will automatically redeploy** (if auto-deploy is enabled)
4. **Or manually redeploy** in Railway dashboard

## Why This Works

- `opencv-python` = Full OpenCV with GUI support (needs display)
- `opencv-python-headless` = OpenCV without GUI (works in servers/cloud)

Both have the same functionality for video/image processing - headless just doesn't require a display.

## After Fix

The app should start successfully! The error will be gone.

