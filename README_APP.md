# Sign Language Recognition - Web App

Simple web application for uploading videos and getting predictions from the trained model.

## 🚀 Quick Start

### 1. Train Model (if needed)

```bash
# Via Colab (recommended)
# Open notebooks/SignLanguage_Training.ipynb in Colab

# Or locally
python scripts/train_model.py --csv Data/Labels/dataset.csv
```

### 2. Run the App

```bash
python app.py
```

### 3. Open in Browser

```
http://localhost:5000
```

**That's it!** Upload a video and get translation! 🎉

---

## 📋 Features

✅ **Simple Web Interface** - Drag and drop videos  
✅ **Multiple Format Support** - MP4, AVI, MOV, MKV, WEBM  
✅ **Video Preview** - Preview video before upload  
✅ **API** - Can also be used via API  
✅ **Modern Design** - Beautiful and user-friendly UI  

---

## 📖 Additional Guides

- **[Detailed Guide](docs/APP_GUIDE.md)** - All details about the app
- **[Training Guide](docs/README_MODEL.md)** - How to train the model
- **[Colab Guide](docs/COLAB_UPLOAD_GUIDE.md)** - Training on Colab

---

**Good luck! 🚀**
