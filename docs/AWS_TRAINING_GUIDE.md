# AWS Training Guide - Sign Language Recognition

מדריך זה מסביר כיצד לאמן את מודל ה-GRU על AWS.

## אפשרויות אימון ב-AWS

יש שתי אפשרויות עיקריות:

### אפשרות 1: Amazon SageMaker (מומלץ)
- ניהול אוטומטי של סביבת ההרצה
- תמיכה מובנית ב-GPU
- ניטור והשוואת ניסויים
- תמחור לפי שימוש

### אפשרות 2: Amazon EC2 עם GPU
- שליטה מלאה על הסביבה
- זול יותר לשימוש ארוך טווח
- דורש הגדרה ידנית

---

## אפשרות 1: Amazon SageMaker

### שלב 1: הכנת נתונים

1. **צור ארכיון של הנתונים:**
```bash
python aws_setup.py --create-archive
```

2. **העלה ל-S3:**
```bash
python aws_setup.py --upload YOUR_BUCKET_NAME --s3-key data/sign_language_data.tar.gz
```

או ידנית:
```bash
aws s3 cp sign_language_data.tar.gz s3://YOUR_BUCKET_NAME/data/
```

### שלב 2: הכנת הקוד

1. **צור סקריפט אימון ל-SageMaker** (`train_sagemaker.py`):
```python
import subprocess
import sys

if __name__ == "__main__":
    # SageMaker sets environment variables
    # Training data is in /opt/ml/input/data/training
    # Model output goes to /opt/ml/model
    
    subprocess.check_call([
        sys.executable, "train_model.py",
        "--csv", "/opt/ml/input/data/training/Data/Labels/dataset.csv",
        "--keypoints-dir", "/opt/ml/input/data/training/Data/Keypoints/rawVideos",
        "--output-dir", "/opt/ml/model",
        "--batch-size", "32",
        "--epochs", "100"
    ])
```

2. **צור Dockerfile** (אופציונלי, או השתמש ב-SageMaker built-in containers):
```dockerfile
FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /opt/ml/code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "train_sagemaker.py"]
```

3. **הרץ אימון ב-SageMaker** (Python SDK):
```python
import sagemaker
from sagemaker.tensorflow import TensorFlow

role = sagemaker.get_execution_role()
sess = sagemaker.Session()

estimator = TensorFlow(
    entry_point='train_sagemaker.py',
    source_dir='.',
    role=role,
    instance_type='ml.p3.2xlarge',  # GPU instance
    instance_count=1,
    framework_version='2.13.0',
    py_version='py39',
    hyperparameters={
        'batch-size': 32,
        'epochs': 100,
        'gru-units': 128
    }
)

# Start training
estimator.fit({'training': 's3://YOUR_BUCKET/data/'})
```

---

## אפשרות 2: Amazon EC2 עם GPU

### שלב 1: הפעל Instance עם GPU

1. **הרץ EC2 Instance:**
   - Instance Type: `g4dn.xlarge` או `g5.xlarge` (GPU instances)
   - AMI: Deep Learning AMI (Ubuntu) - מכיל TensorFlow ו-CUDA מוכנים
   - Storage: לפחות 50GB

2. **התחבר ל-Instance:**
```bash
ssh -i your-key.pem ubuntu@YOUR_INSTANCE_IP
```

### שלב 2: הגדר סביבה

1. **העתק את הפרויקט:**
```bash
# On your local machine
scp -r -i your-key.pem . ubuntu@YOUR_INSTANCE_IP:~/signlanguage/

# Or use git
git clone YOUR_REPO_URL
```

2. **התקן תלויות:**
```bash
cd signlanguage
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **הכן נתונים:**
```bash
# אם הנתונים ב-S3:
aws s3 cp s3://YOUR_BUCKET/data/sign_language_data.tar.gz ./
tar -xzf sign_language_data.tar.gz

# או העתק מקומית:
# scp -r -i your-key.pem Data/ ubuntu@YOUR_INSTANCE_IP:~/signlanguage/
```

### שלב 3: הרץ אימון

```bash
python train_model.py \
    --csv Data/Labels/dataset.csv \
    --keypoints-dir Data/Keypoints/rawVideos \
    --output-dir models \
    --batch-size 32 \
    --epochs 100 \
    --gru-units 128 \
    --num-gru-layers 2 \
    --dropout 0.3 \
    --learning-rate 0.001 \
    --patience 10
```

### שלב 4: העתק מודל מוכן

```bash
# מ-EC2 ל-S3:
aws s3 sync models/ s3://YOUR_BUCKET/models/

# או מ-EC2 למחשב המקומי:
scp -r -i your-key.pem ubuntu@YOUR_INSTANCE_IP:~/signlanguage/models/ ./
```

---

## תמחור משוער

### SageMaker:
- `ml.p3.2xlarge` (GPU): ~$3/שעה
- אימון של ~2-4 שעות: $6-$12

### EC2:
- `g4dn.xlarge` (GPU): ~$0.50/שעה
- `g5.xlarge` (GPU): ~$1.00/שעה
- אימון של ~2-4 שעות: $1-$4

**הערה:** זכור לכבות את ה-instance לאחר האימון!

---

## טיפים

1. **השתמש ב-Spot Instances** (EC2) לחיסכון בעלויות - עד 90% הנחה
2. **שמור את המודל ב-S3** מיד לאחר האימון
3. **השתמש ב-screen או tmux** ב-EC2 כדי שהאימון ימשיך גם אם ההתחברות נקטעת
4. **ניטור:** השתמש ב-CloudWatch לניטור שימוש במשאבים

---

## פקודות שימושיות

```bash
# בדוק GPU availability
nvidia-smi

# הרץ אימון ברקע עם screen
screen -S training
python train_model.py ...
# Ctrl+A, D to detach
screen -r training  # to reattach

# בדוק שימוש בדיסק
df -h

# בדוק שימוש ב-RAM
free -h
```

