# מדריך מלא - Sign Language Recognition Project

מדריך מקיף מקצה לקצה: מההתחלה ועד אימון מודל ב-AWS.

---

## תוכן עניינים

1. [סקירה כללית של הפרויקט](#1-סקירה-כללית-של-הפרויקט)
2. [הכנת הסביבה המקומית](#2-הכנת-הסביבה-המקומית)
3. [חילוץ Keypoints](#3-חילוץ-keypoints)
4. [יצירת Dataset](#4-יצירת-dataset)
5. [הכנה ל-AWS](#5-הכנה-ל-aws)
6. [הגדרת AWS - שלב אחר שלב](#6-הגדרת-aws---שלב-אחר-שלב)
7. [אימון ב-AWS EC2](#7-אימון-ב-aws-ec2)
8. [הורדת המודל](#8-הורדת-המודל)
9. [שימוש במודל](#9-שימוש-במודל)

---

## 0. פתרון אלטרנטיבי - Google Colab (אם AWS Quota נדחה)

**אם AWS דחו את הבקשה ל-quota ואתה רוצה להתחיל מיד:**

✅ **Google Colab** - פתרון מעולה!
- GPU T4 חינם
- מתחיל מיד (אין quota)
- TensorFlow מותקן
- https://colab.research.google.com

📖 ראה `ALTERNATIVES_TO_AWS.md` למדריך מפורט

---

## 1. סקירה כללית של הפרויקט

### מה הפרויקט עושה?

**מטרה**: לזהות מילים בשפת סימנים מסרטונים ולתרגם אותן למילים באנגלית.

### איך זה עובד?

```
┌─────────────────────────────────────────────────────────┐
│                    תהליך מלא                            │
└─────────────────────────────────────────────────────────┘

1. סרטונים (MP4)
   ↓
2. MediaPipe → חילוץ keypoints (21 נקודות לכל יד)
   ↓
3. נרמול → לא תלוי במיקום, גודל, צד היד
   ↓
4. Dataset → CSV עם train/val/test splits
   ↓
5. אימון GRU → מודל לזיהוי sequences
   ↓
6. חיזוי → סרטון חדש → מילה באנגלית
```

### טכנולוגיות:

- **MediaPipe**: חילוץ keypoints מהידיים
- **TensorFlow/Keras**: אימון מודל GRU
- **AWS EC2**: אימון על GPU (זול עם Spot Instances)
- **Python**: כל הקוד

---

## 2. הכנת הסביבה המקומית

### 2.1 התקנת Python

```bash
# בדוק שיש Python 3.8+
python --version

# אם לא, הורד מ-python.org
```

### 2.2 יצירת Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2.3 התקנת תלויות

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**תלויות עיקריות:**
- `opencv-python` - עיבוד וידאו
- `mediapipe` - זיהוי ידיים
- `tensorflow` - אימון מודל
- `numpy`, `pandas`, `scikit-learn` - עיבוד נתונים
- `boto3` - עבודה עם AWS

---

## 3. חילוץ Keypoints

### 3.1 הכנת סרטונים

הניח את הסרטונים בתיקייה:
```
Data/rawVideos/
├── Hello/
│   ├── Hello01.mp4
│   ├── Hello02.mp4
│   └── ...
├── Yes/
├── No/
└── ...
```

### 3.2 הרצת חילוץ

```bash
python scripts/extract_keypoints.py
```

**מה קורה:**
1. הסקריפט עובר על כל הסרטונים
2. MediaPipe מזהה ידיים בכל frame
3. מפיק 21 keypoints לכל יד (wrist, fingers, וכו')
4. מנרמל את ה-keypoints:
   - Wrist ב-(0,0,0) - לא תלוי במיקום
   - Scale לפי גודל היד - לא תלוי בגודל
   - Mirror left/right - לא תלוי בצד היד
   - Rotation alignment - לא תלוי בכיוון
5. שומר כ-`.npy` files ב-`Data/Keypoints/rawVideos/`

**פורמט נתונים:**
- כל סרטון → קובץ `.npy`
- צורה: `(num_frames, 2, 21, 3)`
  - `num_frames`: מספר frames
  - `2`: מספר ידיים
  - `21`: keypoints לכל יד
  - `3`: קואורדינטות (x, y, z)

**זמן**: תלוי במספר הסרטונים, בערך 1-2 דקות לסרטון

### 3.3 בדיקת תוצאות

```bash
# בדוק כמה קבצים נוצרו
python -c "from pathlib import Path; files = list(Path('Data/Keypoints/rawVideos').rglob('*.npy')); print(f'Total: {len(files)}')"
```

---

## 4. יצירת Dataset

### 4.1 יצירת CSV

```bash
python scripts/create_dataset_csv.py
```

**מה קורה:**
1. מוצא את כל קבצי `.npy`
2. מחלק ל-train/val/test (60%/20%/20%)
3. יוצר `Data/Labels/dataset.csv`

**פורמט CSV:**
```csv
path,label,split
keypoints/GoodBye/goodbye02.npy,GOODBYE,train
keypoints/Hello/hello01.npy,HELLO,test
...
```

**תוצאה**: קובץ CSV עם כל הנתונים מחולקים ל-train/val/test

---

## 5. הכנה ל-AWS

### 5.1 יצירת ארכיון נתונים

```bash
python scripts/aws_setup.py --create-archive
```

או ידנית:
```bash
tar -czf sign_language_data.tar.gz Data/
```

**תוצאה**: `sign_language_data.tar.gz` - ארכיון של כל הנתונים

### 5.2 התקנת AWS CLI

**Windows:**
```bash
# הורד מ-aws.amazon.com/cli
# או דרך Chocolatey:
choco install awscli
```

**Linux/Mac:**
```bash
pip install awscli
```

### 5.3 הגדרת AWS Credentials

**חשוב: Root Account vs IAM User**

**לפרויקט אישי:**
- ✅ Root Account יכול לעבוד (אבל לא מומלץ)
- ✅ **מומלץ**: ליצור IAM User (5 דקות, יותר בטוח)
- 📖 ראה `AWS_SECURITY_GUIDE.md` למדריך מפורט

**יצירת IAM User (מומלץ):**
1. AWS Console → IAM → Users → Create user
2. שם: `sign-language-user`
3. Permissions: `AmazonEC2FullAccess` + `AmazonS3FullAccess`
4. Create access key → שמור את ה-Keys

**⚠️ הערה: אם אתה משתמש ב-AMIs מ-Marketplace:**
- ייתכן שתצטרך להוסיף `AWSMarketplaceFullAccess` permission
- **או פשוט בחר AMI מ-Community AMIs** - אין צורך ב-Subscribe!

**הגדרת AWS CLI:**

```bash
aws configure
```

**תצטרך:**
- AWS Access Key ID (מ-Root או מ-IAM User)
- AWS Secret Access Key
- Default region (למשל: `us-east-1`)
- Default output format: `json`

**איך להשיג Access Keys:**

**אם משתמש ב-Root:**
1. היכנס ל-AWS Console
2. לחץ על השם שלך (ימין למעלה) → "Security credentials"
3. לחץ "Create access key"
4. שמור את ה-Access Key ID וה-Secret Access Key (תראה רק פעם אחת!)

**אם משתמש ב-IAM User (מומלץ):**
1. AWS Console → IAM → Users → בחר את ה-user
2. Security credentials → Access keys → Create access key
3. שמור את ה-Keys

---

## 6. הגדרת AWS - שלב אחר שלב

### 6.1 יצירת S3 Bucket

**ב-AWS Console:**

1. **היכנס ל-AWS Console**: https://console.aws.amazon.com
2. **חפש "S3"** בשורת החיפוש העליונה
3. **לחץ "Create bucket"**
4. **מלא פרטים:**
   - **Bucket name**: `sign-language-project-yourname` (חייב להיות ייחודי)
   - **Region**: בחר region קרוב (למשל: `us-east-1`)
   - **Object Ownership**: השאר ברירת מחדל
   - **Block Public Access**: השאר מופעל (אבטחה)
5. **לחץ "Create bucket"**

**או דרך CLI:**
```bash
aws s3 mb s3://sign-language-project-yourname --region us-east-1
```

### 6.2 העלאת נתונים ל-S3

**דרך CLI (מומלץ):**
```bash
# העלה את הארכיון
aws s3 cp sign_language_data.tar.gz s3://sign-language-project-yourname/data/

# או העלה את כל תיקיית Data
aws s3 sync Data/ s3://sign-language-project-yourname/data/
```

**דרך Console:**
1. היכנס ל-S3 → בחר את ה-bucket
2. לחץ "Upload"
3. גרור את `sign_language_data.tar.gz` או `Data/`
4. לחץ "Upload"

**זמן**: תלוי בגודל, בערך 5-10 דקות

### 6.3 יצירת Key Pair (להתחברות ל-EC2)

**ב-AWS Console:**

1. **חפש "EC2"** בשורת החיפוש
2. **בסרגל השמאלי → "Key Pairs"** (תחת "Network & Security")
3. **לחץ "Create key pair"**
4. **מלא פרטים:**
   - **Name**: `sign-language-key` (או שם אחר)
   - **Key pair type**: RSA
   - **Private key file format**: `.pem` (ל-Windows/Linux) או `.ppk` (ל-PuTTY)
5. **לחץ "Create key pair"**
6. **הורד את הקובץ** - שמור אותו במקום בטוח! (תראה רק פעם אחת)

**חשוב**: שמור את הקובץ `.pem` - תצטרך אותו להתחברות!

### 6.4 הפעלת EC2 Instance

**ב-AWS Console:**

#### שלב 1: Launch Instance

1. **ב-EC2 Console → לחץ "Launch Instance"** (כפתור גדול)
2. **מלא שם**: `Sign Language Training` (אופציונלי)

#### שלב 2: בחר AMI (Amazon Machine Image)

**חשוב: אם אתה משתמש ב-IAM User, יש שתי אפשרויות:**

**אפשרות 1: בחר AMI מ-AWS הרשמי או Community (מומלץ - לא צריך Subscribe):**

1. **לחץ "Browse more AMIs"**
2. **בחר את הטאב "Community AMIs"** או **"Quick Start"** (למעלה)
3. **חפש AMI תומך GPU:**
   
   **אופציות מומלצות:**
   - ✅ **"Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)"** - מ-AWS הרשמי! ✅
     - Verified provider (AWS)
     - Ubuntu 22.04
     - CUDA מותקן
     - **בחר: 64-bit (x86)** - לא Arm!
   - ✅ `Deep Learning Base GPU AMI Ubuntu` - מ-Community
   - ✅ `Deep Learning Base GPU AMI (Ubuntu 20.04)` - גם טוב
   - ✅ כל AMI עם "GPU AMI" + "Ubuntu" בשם
   
   **חשוב להבין:**
   - ⚠️ **ה-AMI עצמו חינמי!** (כל ה-AMIs חינמיים)
   - 💰 **מה שכן עולה זה ה-EC2 instance** (GPU) - ~$0.05-0.50/שעה
   - 💰 **המחיר לא תלוי ב-AMI - תלוי ב-instance type!**
   
4. **בחר:** AMI עם "GPU" או "CUDA" בשם ו-Ubuntu
5. **אם יש בחירת Architecture: בחר "64-bit (x86)"** - לא Arm!
6. **אין צורך ב-Subscribe!** ✅

**⚠️ הערה חשובה:**
- כל ה-AMIs חינמיים (לא עולים כסף)
- מה שעולה זה רק ה-EC2 instance עם GPU
- המחיר תלוי ב-instance type (g4dn.xlarge, g5.xlarge, וכו') - לא ב-AMI
- כל AMI תומך GPU יעלה אותו מחיר (תלוי ב-instance type)

**אפשרות 2: AMI מ-Marketplace (דורש Subscribe):**

⚠️ **אם תבחר AMI מ-Marketplace:**
- תראה שגיאה: "Instance launch failed. An error occurred while attempting to subscribe to this AMI"
- או: "not authorized to perform: aws-marketplace:Subscribe"

**פתרונות:**

**פתרון 1: בחר AMI מ-Community במקום (מומלץ!)** ✅
- פשוט חזור לבחירת AMI
- בחר "Community AMIs" tab
- אין צורך ב-Subscribe!

**פתרון 2: Subscribe ידנית עם Root Account:**
1. היכנס ל-AWS Console עם Root Account
2. לך ל-URL מהשגיאה: `https://aws.amazon.com/marketplace/pp?sku=...`
3. לחץ "Subscribe" או "Continue to Subscribe"
4. אחרי Subscribe, חזור ל-EC2 ונסה שוב

**פתרון 3: הוסף permissions ל-IAM User:**
1. AWS Console → IAM → Users → בחר `sign-language-user`
2. Add permissions → Attach policies directly
3. חפש: `AWSMarketplaceFullAccess`
4. סמן ובחר "Next" → "Add permissions"

**אם אתה משתמש ב-Root Account:**
- כל ה-AMIs יעבדו (גם Marketplace)

**המלצה:** בחר AMI מ-Community AMIs - פשוט יותר ואין צורך ב-Subscribe!

#### שלב 3: בחר Instance Type

1. **לחץ "Instance types"**
2. **סנן לפי:**
   - **GPU instances** (g4dn, g5, וכו')
3. **בחר:**
   - **g4dn.xlarge** - מומלץ (1 GPU, 4 vCPU, 16GB RAM) - $0.50/שעה
   - **g4dn.2xlarge** - יותר כוח (1 GPU, 8 vCPU, 32GB RAM) - $0.75/שעה
   - **g5.xlarge** - GPU חדש יותר - $1.00/שעה

**לחיסכון - Spot Instances:**
1. **לחץ על "Configure instance"** (למטה)
2. **בחלק "Purchasing option"**:
   - סמן ☑ **"Request Spot Instances"**
   - **Max price**: השאר "On-demand price" (בטוח) או 70% מהמחיר (יותר זול)
   - **Request type**: Persistent
3. **חיסכון**: עד 90% הנחה! ($0.05/שעה במקום $0.50)

⚠️ **אם אתה רואה שגיאה: "Max spot instance count exceeded":**
- זה אומר שהגעת למגבלה של Spot instances
- **פתרון 1:** השתמש ב-On-Demand במקום (None) - יקר יותר אבל יעבוד
- **פתרון 2:** בדוק כמה instances יש לך רץ - אולי יש לך instances ישנים שצריך לסגור
- **פתרון 3:** חכה קצת (10-30 דקות) ונסה שוב
- **פתרון 4:** בקש להגדיל Service Quota (Service Quotas → EC2 → Spot Instances)

#### שלב 4: הגדר Key Pair

1. **ב-"Key pair (login)"**:
   - בחר את ה-Key Pair שיצרת קודם (`sign-language-key`)
2. **ודא**: "Create a new key pair" לא מסומן (אם כבר יש לך)

#### שלב 5: הגדר Network Settings

1. **Security groups**: בחר "Create security group"
2. **פתח פורטים:**
   - ✅ **SSH (22)**: מהכתובת שלך בלבד (My IP) - **חובה!**
   - ❌ **אל תפתח CUSTOMTCP עם 0.0.0.0/0** - זה לא בטוח!
   - ❌ **אל תפתח HTTP/HTTPS** - לא צריך לפרויקט
   
**חשוב - אבטחה:**
- פתח **רק SSH (22) מ-My IP** - זה כל מה שצריך לאימון
- אם יש "Allow CUSTOMTCP traffic from Anywhere (0.0.0.0/0)" - **בטל את הסימון!**
- זה לא בטוח לפתוח גישה מכל מקום

#### שלב 6: הגדר Storage

1. **Configure storage**:
   - **Size**: 50GB (מינימום) או 100GB (מומלץ)
   - **Volume type**: gp3 (SSD) - מומלץ

#### שלב 7: Launch!

1. **לחץ "Launch Instance"** (כחול, למטה)
2. **המתן 2-5 דקות** עד שה-instance יעלה
3. **לחץ "View all instances"** כדי לראות את הסטטוס

#### שלב 8: בדוק את ה-Instance

1. **ב-EC2 Dashboard → Instances**
2. **חכה עד ש-"Instance state" = "Running"** (ירוק)
3. **חכה עד ש-"Status checks" = "2/2 checks passed"**
4. **שמור את ה-IPv4 Public IP** (למשל: `54.123.45.67`)

**חשוב**: שמור את ה-IP - תצטרך אותו להתחברות!

---

## 7. אימון ב-AWS EC2

### 7.1 התחברות ל-EC2

**Windows (PowerShell):**

```powershell
# אם יש בעיית permissions:
icacls your-key.pem /inheritance:r
icacls your-key.pem /grant:r "%username%:R"

# התחבר
ssh -i sign-language-key.pem ubuntu@YOUR_INSTANCE_IP
```

**Linux/Mac:**

```bash
# שנה permissions
chmod 400 sign-language-key.pem

# התחבר
ssh -i sign-language-key.pem ubuntu@YOUR_INSTANCE_IP
```

**אם זה עובד**: תראה משהו כמו:
```
Welcome to Ubuntu...
ubuntu@ip-xxx-xxx-xxx-xxx:~$
```

**אתה עכשיו בתוך המחשב ב-AWS!**

### 7.2 בדיקת GPU

```bash
nvidia-smi
```

**אמור להציג:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ... Driver Version: ...                                        |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   30C    P0    26W /  70W |      0MiB / 15109MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

**אם אתה רואה GPU** - הכל תקין! ✅

### 7.3 העתקת הפרויקט

**אפשרות 1: Git (אם יש repository)**

```bash
cd ~
git clone YOUR_REPO_URL
cd signlanguage
```

**אפשרות 2: SCP מהמחשב המקומי**

**מהמחשב המקומי (PowerShell):**
```powershell
# העתק את כל הפרויקט
scp -i sign-language-key.pem -r . ubuntu@YOUR_INSTANCE_IP:~/signlanguage/
```

**אפשרות 3: S3 (אם העלית קודם)**

```bash
# ב-EC2
cd ~
aws s3 sync s3://sign-language-project-yourname/code/ ./signlanguage/
```

### 7.4 הורדת נתונים מ-S3

```bash
# ב-EC2
cd ~/signlanguage

# הורד את הנתונים
aws s3 sync s3://sign-language-project-yourname/data/Data/ ./Data/

# או אם העלית ארכיון:
aws s3 cp s3://sign-language-project-yourname/data/sign_language_data.tar.gz ./
tar -xzf sign_language_data.tar.gz
```

### 7.5 התקנת תלויות

```bash
cd ~/signlanguage

# Deep Learning AMI כבר מכיל Python, אבל בואו נוודא
python3 --version

# צור virtual environment
python3 -m venv venv
source venv/bin/activate

# התקן תלויות
pip install --upgrade pip
pip install -r requirements.txt
```

**זמן**: 5-10 דקות

### 7.6 התקנת screen (חשוב!)

```bash
sudo apt-get update
sudo apt-get install screen -y
```

**למה screen?**
- האימון יכול לקחת שעות
- אם ההתחברות תתנתק, האימון ימשיך לרוץ
- אפשר להתחבר מחדש ולבדוק התקדמות

### 7.7 הרצת אימון

```bash
# צור screen session
screen -S training

# בתוך screen:
cd ~/signlanguage
source venv/bin/activate

# הרץ אימון
python scripts/train_model.py \
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

**ניתוק מ-screen:**
- לחץ `Ctrl+A` ואז `D` (detach)
- Session ימשיך לרוץ ברקע!

**התחברות מחדש:**
```bash
screen -r training
```

### 7.8 שמירת מודל ל-S3 (תכופות!)

**בתוך screen, או ב-terminal נפרד:**

```bash
# שמור את המודל הטוב ביותר
aws s3 sync models/ s3://sign-language-project-yourname/models/ \
    --exclude "*" \
    --include "*.keras" \
    --include "*.json"
```

**מומלץ**: לעשות את זה כל epoch או כל 10 epochs.

**או ב-cron job (אוטומטי):**
```bash
# ערוך crontab
crontab -e

# הוסף שורה (כל 30 דקות):
*/30 * * * * cd ~/signlanguage && aws s3 sync models/ s3://sign-language-project-yourname/models/ --exclude "*" --include "*.keras" --include "*.json"
```

### 7.9 ניטור האימון

**בתוך screen:**
- תראה את ההתקדמות בזמן אמת
- Loss ו-Accuracy מתעדכנים

**מחוץ ל-screen:**
```bash
# בדוק אם process רץ
ps aux | grep python

# בדוק GPU usage
watch -n 1 nvidia-smi

# בדוק disk space
df -h

# בדוק memory
free -h
```

---

## 8. הורדת המודל

### 8.1 מהמחשב המקומי

```bash
# הורד את המודל מ-S3
aws s3 sync s3://sign-language-project-yourname/models/ ./models/
```

**תוצאה**: המודל נמצא ב-`models/run_TIMESTAMP/best_model.keras`

### 8.2 או דרך SCP

```bash
# מהמחשב המקומי
scp -i sign-language-key.pem -r ubuntu@YOUR_INSTANCE_IP:~/signlanguage/models/ ./models/
```

---

## 9. שימוש במודל

### 9.1 חיזוי מסרטון

```bash
python scripts/predict.py \
    --model models/run_YYYYMMDD_HHMMSS/best_model.keras \
    --video path/to/video.mp4
```

**תוצאה:**
```
Prediction Results
============================================================
Predicted: HELLO
Confidence: 0.8542 (85.42%)

Top 3 predictions:
  1. HELLO: 0.8542 (85.42%)
  2. YES: 0.1023 (10.23%)
  3. NO: 0.0234 (2.34%)
============================================================
```

### 9.2 חיזוי מ-keypoints

```bash
python scripts/predict.py \
    --model models/run_YYYYMMDD_HHMMSS/best_model.keras \
    --keypoints Data/Keypoints/rawVideos/Hello/Hello01.npy
```

---

## 10. כיבוי Instance (חשוב מאוד!)

### דרך Console:

1. **ב-EC2 Dashboard → Instances**
2. **בחר את ה-instance**
3. **לחץ "Instance state" → "Stop instance"** (להפסקה זמנית)
   - או **"Terminate instance"** (למחיקה מלאה)
4. **אשר**

### דרך CLI:

```bash
# Stop (אפשר להפעיל מחדש)
aws ec2 stop-instances --instance-ids i-xxxxxxxxxxxxx

# Terminate (מוחק לגמרי)
aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxxx
```

**חשוב מאוד**: כבה את ה-instance כשסיימת כדי לא לשלם!

---

## 11. עלויות וניהול

### 11.1 עלויות - סיכום מהיר

**חשוב: IAM User חינם!** ✅

**עלויות הפרויקט:**

| תרחיש | Spot Instances | On-Demand |
|-------|----------------|-----------|
| אימון אחד (4 שעות) | **$0.20-0.40** | $2.00-4.00 |
| מספר אימונים (10 שעות) | **$0.50-1.00** | $5.00-10.00 |
| S3 Storage (חודש) | **$0.05-0.10** | $0.05-0.10 |

**עם Spot Instances: הפרויקט יכול לעלות פחות מדולר!** 💰

📖 ראה `AWS_COSTS.md` לחישוב מפורט

### 11.2 בדיקת עלויות

**ב-AWS Console:**

1. **חפש "Billing"** או "Cost Management"
2. **Cost Explorer** - ראה עלויות לפי זמן
3. **Bills** - ראה חשבוניות

### 11.3 הגדרת Billing Alerts

1. **Billing → Preferences → Billing alerts**
2. **Create alert**:
   - **Alert threshold**: $10 (או סכום אחר)
   - **Email**: כתובת שלך
3. **תקבל email** אם העלויות עוברות את הסף

**חשוב מאוד**: הגדר alerts כדי לדעת אם אתה משלם יותר מדי!

---

## 12. Troubleshooting

### בעיות נפוצות

#### 1. לא יכול להתחבר ל-EC2

**פתרונות:**
- בדוק שה-Key Pair נכון
- בדוק שה-Security Group מאפשר SSH
- בדוק שה-Instance Running
- נסה `ssh -v` לראות שגיאות

#### 2. GPU לא מזוהה

```bash
# בדוק drivers
nvidia-smi

# אם לא עובד:
sudo apt-get update
sudo apt-get install -y nvidia-driver-470
sudo reboot
```

#### 3. Out of Memory

```bash
# הקטן batch size
--batch-size 16

# או השתמש ב-gradient accumulation
```

#### 4. Connection נקטע

```bash
# השתמש ב-screen
screen -S training
# הרץ אימון
# Ctrl+A, D לניתוק
```

#### 5. שגיאה: "vCPU limit of 0 allows" או "You have requested more vCPU capacity"

**מה זה אומר:**
- אתה חדש ב-AWS ולא הוגדר לך vCPU limit ל-GPU instances
- vCPU limit של 0 = לא יכול להפעיל GPU instances
- ⚠️ **זה הכרחי - אין דרך אחרת!** צריך לבקש הגדלה

**⚠️ חשוב להבין:**
- **אין חלופה** - אתה חייב quota ל-GPU instances
- זה תהליך חד-פעמי (אחרי אישור, זה לתמיד)
- זה לוקח כמה שעות, אבל זה הכרחי
- **זה חינם** - רק מבקשים הרשאה, לא משלמים כסף

**פתרון: בקש הגדלת vCPU limit (זה הכרחי!)**

**דרך 1: דרך URL ישיר (הכי פשוט):**

1. **פתח את ה-URL מהשגיאה:**
   ```
   http://aws.amazon.com/contact-us/ec2-request
   ```

2. **אם אתה רואה טופס Support:**
   - ✅ **לחץ על הלינק הכחול**: "Looking for service quota increases?" (מימין)
   - זה יקח אותך ישירות לטופס הנכון!

3. **אם אתה בטופס Quota Increase:**
   - **Service**: EC2
   - **Region**: בחר את ה-region שלך (למשל: US East (N. Virginia) - us-east-1)
   - **Limit type**: Running On-Demand G instances (או "G and VT instances")
   - **Instance type**: g4dn.xlarge
   - **New limit value**: 4
   - **Use case**: "Machine Learning / Deep Learning training"
   - **Description**: "Need GPU instances for deep learning model training. Training sign language recognition model using TensorFlow/Keras."

4. **Submit**

**אם אתה נשאר בטופס Support (לא מומלץ):**
- **Issue type**: Technical (לא Account and billing)
- **Service**: EC2
- **Category**: Service Limits
- **Severity**: Normal
- אבל **עדיף ללחוץ על "Looking for service quota increases?"**

**דרך 2: דרך Service Quotas (מומלץ - דרך החלון):**

1. **לחץ על הכפתור הכתום: "Service Quotas dashboard"**
   - זה יקח אותך ישירות ל-Service Quotas

2. **בחר Region** (למעלה, למשל: us-east-1)

3. **חפש את ה-Quota הנכון:**
   
   **דרך 1: חיפוש (מומלץ):**
   - **בתיבת החיפוש "Search by quota name"**, הזן: `Running On-Demand G`
   - או: `G instances`
   - לחץ Enter
   
   **דרך 2: גלול ברשימה:**
   - גלול למטה ברשימה
   - חפש: **"Running On-Demand G instances"**
   - או: **"Running On-Demand G and VT instances"** ✅ **זה מה שאתה צריך!**
   - זה ל-GPU instances (g4dn, g5, וכו')
   
   **✅ "Running On-Demand G and VT instances" זה בדיוק מה שאתה צריך!**
   - G instances = GPU instances (g4dn, g5, וכו')
   - VT instances = GPU instances מסוג אחר
   - זה בדיוק ל-GPU instances שלך!
   
   **⚠️ חשוב:**
   - אל תבחר "All G and VT Spot Instance Requests" - זה ל-Spot, לא מה שאתה צריך!
   - אתה צריך "Running On-Demand G and VT instances" ✅

4. **לחץ על ה-Quota**

5. **Request quota increase:**
   
   **בשדה "Increase quota value":**
   - **שנה את הערך ל-4** (במקום 2)
   - g4dn.xlarge צריך 4 vCPUs
   - אם אתה רוצה g4dn.2xlarge, תשים 8
   
   **אם יש שדות נוספים:**
   - **Use case**: "Machine Learning / Deep Learning"
   - **Description**: "Need GPU instances for deep learning model training. Training sign language recognition model using TensorFlow/Keras."
   
   **אבל אם אין שדות נוספים - זה בסדר, רק תשנה ל-4**

6. **לחץ על הכפתור הכתום "Request"** (ימין למטה)

**או דרך ישירה:**
- AWS Console → חפש "Service Quotas"
- EC2 → Running On-Demand G instances
- Request quota increase

**זמן המתנה - כמה זמן עד אישור:**

**לבקשה קטנה (4 vCPUs - מה שאתה מבקש):**
- ⏱️ **בדרך כלל: 2-6 שעות**
- ⏱️ **מקסימום: 24 שעות**
- ✅ **רוב הבקשות הקטנות מאושרות תוך כמה שעות**

**לבקשה גדולה (יותר מ-16 vCPUs):**
- ⏱️ **יכול לקחת יותר זמן: 24-48 שעות**
- ⏱️ **לפעמים צריך אישור ידני**

**מה קורה אחרי שליחת הבקשה:**
1. תקבל email אישור שהבקשה התקבלה
2. AWS בודק את הבקשה (אוטומטי או ידני)
3. תקבל email כשזה מאושר (או נדחה)
4. ב-Service Quotas תראה את הסטטוס: "Pending" → "Approved"

**איך לבדוק סטטוס:**
- Service Quotas → Request history
- תראה את כל הבקשות והסטטוס שלהן

**טיפ:**
- בדוק את ה-email שלך - תקבל עדכון כשזה מאושר
- אפשר גם לבדוק ב-Service Quotas → Request history

**⚠️ אם הבקשה נדחתה:**

**מה לעשות:**
1. **פתח את ה-case מחדש** (Reopen case)
   - ב-email יש case number (למשל: CASE 1767286722008921)
   - לך ל-AWS Support → Cases → פתח את ה-case

2. **ספק use case מפורט:**
   - הסבר מה אתה עושה: "Training deep learning model for sign language recognition"
   - הסבר למה אתה צריך GPU: "Model requires GPU for training (TensorFlow/Keras with GRU)"
   - הסבר על הפרויקט: "Academic/research project for sign language translation"
   - הסבר על העלויות: "Using Spot Instances to minimize costs"
   - הסבר על השימוש: "One-time training session, will terminate instance after training"

3. **Submit מחדש**

**דוגמה ל-use case מפורט:**
```
I am working on a deep learning project for sign language recognition. 
I need to train a GRU (Gated Recurrent Unit) neural network model using 
TensorFlow/Keras. The model processes video sequences of hand keypoints 
extracted from sign language videos.

The training requires GPU acceleration (g4dn.xlarge instance) as the 
model processes sequences of 21 hand keypoints per frame across multiple 
frames. Without GPU, training would take days or weeks.

I plan to use Spot Instances to minimize costs (approximately $0.05/hour 
instead of $0.50/hour). The training session will be a one-time event, 
and I will terminate the instance immediately after training completes.

This is for an academic/research project to create a sign language 
translation application. I have already prepared the dataset locally 
and uploaded it to S3. I only need 4 vCPUs for a single g4dn.xlarge 
instance to complete this training.

I understand AWS service quotas and will monitor costs carefully. 
I have set up billing alerts to ensure I don't exceed my budget.
```

**למה זה יכול לעזור:**
- AWS רוצה לראות use case מפורט
- הם רוצים להבין למה אתה צריך את זה
- הם רוצים לראות שאתה מבין עלויות
- Use case מפורט עוזר להם לאשר

**אחרי שאושר:**
1. תקבל email אישור
2. חזור ל-EC2 Console
3. נסה שוב להפעיל instance
4. זה יעבוד! ✅

**❓ שאלות נפוצות:**

**Q: זה עולה כסף?**  
A: לא! זה רק בקשה להרשאה, לא עולה כסף.

**Q: כמה זמן זה לוקח?**  
A: 2-6 שעות בדרך כלל, מקסימום 24 שעות.

**Q: זה חד-פעמי?**  
A: כן! אחרי אישור, זה לתמיד.

**Q: יש דרך אחרת?**  
A: לא. אתה חייב quota ל-GPU instances.

**❓ שאלות נפוצות:**

**Q: האם זה משנה אם אני עושה את הבקשה ב-Root User או ב-IAM User?**  
A: **לא! זה לא משנה כלל.** Quota increases הם ברמת Account, לא ברמת User. כלומר:
- הבקשה תעבוד מ-Root או מ-IAM User
- אחרי אישור, **כל ה-Users** ב-Account יוכלו להשתמש ב-quota החדש
- אז לא משנה איזה user מבקש - זה יפעיל על כל ה-Account

**Q: עדיף ב-Root או ב-IAM User?**  
A: לא משנה, אבל Root User יש לו את כל ה-permissions אז זה בטוח לעבוד.

**⚠️ חשוב:**
- זה **הכרחי** - אין דרך אחרת להפעיל GPU instances
- זה **חינם** - רק מבקשים הרשאה
- זה **חד-פעמי** - אחרי אישור, זה לתמיד
- זה **ברמת Account** - כל ה-Users יכולים להשתמש אחרי אישור

#### 5. Instance נפסק (Spot)

- זה נורמלי! AWS יכול לעצור Spot Instances
- המודל שומר checkpoints אוטומטית
- פשוט הפעל instance חדש והמשך

---

## 13. Workflow מלא - סיכום

```
┌─────────────────────────────────────────────────────────┐
│                    Workflow מלא                         │
└─────────────────────────────────────────────────────────┘

1. מקומי: חילוץ keypoints
   python scripts/extract_keypoints.py
   ↓
2. מקומי: יצירת dataset
   python scripts/create_dataset_csv.py
   ↓
3. מקומי: העלאת נתונים ל-S3
   aws s3 sync Data/ s3://bucket/data/
   ↓
4. AWS Console: הפעלת EC2 instance
   - בחר Deep Learning AMI
   - בחר g4dn.xlarge (Spot)
   - בחר Key Pair
   ↓
5. EC2: התחברות והכנה
   ssh -i key.pem ubuntu@IP
   aws s3 sync s3://bucket/data/ ./Data/
   pip install -r requirements.txt
   ↓
6. EC2: אימון (ב-screen)
   screen -S training
   python scripts/train_model.py ...
   ↓
7. EC2: שמירה ל-S3 (תכופות)
   aws s3 sync models/ s3://bucket/models/
   ↓
8. מקומי: הורדת מודל
   aws s3 sync s3://bucket/models/ ./models/
   ↓
9. מקומי: שימוש
   python scripts/predict.py --model models/.../best_model.keras --video test.mp4
   ↓
10. AWS Console: כיבוי instance
    Stop/Terminate instance
```

---

## 14. טיפים חשובים

### 1. תמיד שמור ל-S3
- כל epoch או כל 10 epochs
- אם instance נפסק, לא תאבד עבודה

### 2. השתמש ב-screen
- האימון יכול לקחת שעות
- screen מאפשר להתנתק ולהתחבר מחדש

### 3. Spot Instances
- חוסך 90% בעלויות
- מומלץ מאוד!

### 4. כבה instance מיד
- כשסיימת, כבה מיד
- אחרת תמשיך לשלם

### 5. הגדר billing alerts
- תדע אם אתה משלם יותר מדי
- תקבל התראה לפני שזה יקר מדי

---

## 15. פקודות שימושיות

### מקומי:

```bash
# חילוץ keypoints
python scripts/extract_keypoints.py

# יצירת dataset
python scripts/create_dataset_csv.py

# העלאת נתונים
aws s3 sync Data/ s3://bucket/data/

# הורדת מודל
aws s3 sync s3://bucket/models/ ./models/

# חיזוי
python scripts/predict.py --model models/.../best_model.keras --video test.mp4
```

### ב-EC2:

```bash
# התחברות
ssh -i key.pem ubuntu@IP

# הורדת נתונים
aws s3 sync s3://bucket/data/ ./Data/

# screen
screen -S training
screen -r training  # התחברות מחדש

# אימון
python scripts/train_model.py --csv Data/Labels/dataset.csv

# שמירה
aws s3 sync models/ s3://bucket/models/

# ניטור
nvidia-smi
df -h
free -h
```

---

## 16. עלויות משוערות

### תרחיש: אימון של 4 שעות

| Instance | On-Demand | Spot | עם Spot |
|----------|-----------|------|---------|
| g4dn.xlarge | $0.50/שעה | $0.05/שעה | **$0.20** |
| g4dn.2xlarge | $0.75/שעה | $0.08/שעה | **$0.32** |
| g5.xlarge | $1.00/שעה | $0.10/שעה | **$0.40** |

**עם Spot: אימון מלא יכול לעלות פחות מדולר!**

### עלויות נוספות:

- **S3 Storage**: ~$0.023/GB/חודש (זניח)
- **Data Transfer**: חינם בתוך region
- **Total**: בעיקר עלות ה-EC2 instance

---

## 17. שאלות נפוצות

**Q: כמה זמן לוקח אימון?**  
A: תלוי בנתונים, בערך 2-6 שעות.

**Q: מה אם instance נפסק (Spot)?**  
A: המודל שומר checkpoints. הפעל instance חדש והמשך.

**Q: צריך ידע טכני?**  
A: בסיסי - SSH, Linux commands. המדריך מפורט.

**Q: מה אם אני שוכח לכבות?**  
A: הגדר billing alerts. תמיד תזכור לכבות!

**Q: איך אני יודע כמה זה עלה?**  
A: AWS Cost Explorer או Billing Dashboard.

**Q: מה אם יש שגיאה באימון?**  
A: בדוק logs, נסה להקטין batch size, או לבדוק את הנתונים.

---

## 18. Next Steps

1. ✅ הכן את הסביבה המקומית
2. ✅ חלץ keypoints מהסרטונים
3. ✅ צור dataset CSV
4. ✅ העלה נתונים ל-S3
5. ✅ הפעל EC2 instance
6. ✅ אמן את המודל
7. ✅ הורד את המודל
8. ✅ השתמש במודל לחיזוי

---

## 19. משאבים נוספים

- **AWS EC2 Documentation**: https://docs.aws.amazon.com/ec2/
- **TensorFlow Documentation**: https://www.tensorflow.org/
- **MediaPipe Documentation**: https://mediapipe.dev/

---

## סיכום

**הפרויקט כולל:**
1. חילוץ keypoints מסרטונים (MediaPipe)
2. נרמול מתקדם (מיקום, גודל, צד, כיוון)
3. אימון מודל GRU (TensorFlow)
4. אימון ב-AWS EC2 (זול עם Spot)
5. חיזוי מסרטונים חדשים

**עלויות:**
- עם Spot Instances: **פחות מדולר לאימון מלא!**

**זמן:**
- הכנה: 1-2 שעות
- אימון: 2-6 שעות
- **סה"כ: יום עבודה אחד**

---

**בהצלחה! 🚀**

אם יש שאלות, ראה את המדריכים המפורטים ב-`docs/` או `IMPLEMENTATION_GUIDE.md`.

