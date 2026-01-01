# AWS EC2 Training Guide - Sign Language Recognition

מדריך מפורט לאימון מודל GRU על Amazon EC2 (לא SageMaker) - פתרון חסכוני.

## למה EC2 ולא SageMaker?

- **עלות נמוכה יותר**: EC2 instances זולים יותר, במיוחד Spot Instances (עד 90% הנחה)
- **שליטה מלאה**: גישה מלאה לסביבה, יכולת להתקין כל מה שצריך
- **גמישות**: אפשר לשנות instance types, להפסיק ולהמשיך, וכו'
- **אידיאלי לפרויקטים**: לפרויקטים קטנים-בינוניים, EC2 יותר כלכלי

## עלויות משוערות

### Instance Types מומלצים:
- **g4dn.xlarge** (GPU): ~$0.50/שעה
- **g5.xlarge** (GPU, חדש יותר): ~$1.00/שעה
- **g4dn.2xlarge** (יותר כוח): ~$0.75/שעה

### Spot Instances (מומלץ!):
- **עד 90% הנחה** - g4dn.xlarge יכול להיות ~$0.05-0.10/שעה
- **אזהרה**: Instance יכול להיפסק אם AWS צריך את המשאבים
- **פתרון**: שמור checkpoints תכופים

### אימון משוער:
- אימון של 2-4 שעות: **$0.20-$4** (תלוי ב-instance type)
- עם Spot: **$0.10-$0.40**

---

## שלב 1: הכנת נתונים מקומית

### 1.1 צור ארכיון של הנתונים

```bash
# צור ארכיון של כל הנתונים
python aws_setup.py --create-archive

# או ידנית:
tar -czf sign_language_data.tar.gz Data/
```

### 1.2 העלה ל-S3

```bash
# צור S3 bucket (אם עדיין לא קיים)
aws s3 mb s3://your-bucket-name

# העלה את הנתונים
aws s3 cp sign_language_data.tar.gz s3://your-bucket-name/data/

# או העלה את כל תיקיית Data
aws s3 sync Data/ s3://your-bucket-name/data/
```

---

## שלב 2: הפעלת EC2 Instance

### 2.1 בחר AMI (Amazon Machine Image)

**מומלץ: Deep Learning AMI (Ubuntu)**
- מכיל TensorFlow, PyTorch, CUDA מוכנים
- חוסך זמן בהתקנה
- AMI ID משתנה לפי region

**איך למצוא:**
1. ב-AWS Console → EC2 → Launch Instance
2. בחר "Deep Learning AMI (Ubuntu)" או "Deep Learning Base AMI"
3. או חפש ב-AMI Marketplace: "Deep Learning"

### 2.2 בחר Instance Type

**עבור אימון GRU:**
- **g4dn.xlarge** - מספיק לרוב המקרים (1 GPU, 4 vCPU, 16GB RAM)
- **g4dn.2xlarge** - אם יש הרבה נתונים (1 GPU, 8 vCPU, 32GB RAM)
- **g5.xlarge** - GPU חדש יותר, קצת יותר יקר

**עבור Spot Instance:**
- סמן "Request Spot Instances"
- בחר "Max price" - 70% מהמחיר הרגיל (בטוח)
- או השאר "On-demand price" (יותר בטוח)

### 2.3 הגדר Storage

- **מינימום 50GB** (יותר טוב 100GB)
- SSD (gp3) מומלץ

### 2.4 הגדר Security Group

פתח פורטים:
- **SSH (22)** - מהכתובת שלך בלבד
- **Jupyter (8888)** - אופציונלי, אם רוצה Jupyter

### 2.5 בחר Key Pair

- בחר או צור Key Pair (`.pem` file)
- **שמור את הקובץ!** - תצטרך אותו להתחברות

---

## שלב 3: התחברות והגדרה

### 3.1 התחבר ל-Instance

```bash
# Windows (PowerShell)
ssh -i your-key.pem ubuntu@YOUR_INSTANCE_IP

# אם יש בעיית permissions:
icacls your-key.pem /inheritance:r
icacls your-key.pem /grant:r "%username%:R"
```

### 3.2 בדוק GPU

```bash
nvidia-smi
```

אמור להציג את ה-GPU שלך.

### 3.3 הורד את הפרויקט

**אפשרות 1: Git (אם יש repository)**
```bash
cd ~
git clone YOUR_REPO_URL
cd signlanguage
```

**אפשרות 2: העתקה מ-S3**
```bash
cd ~
aws s3 cp s3://your-bucket-name/data/sign_language_data.tar.gz ./
tar -xzf sign_language_data.tar.gz

# העתק את הקוד (אם יש לך)
# או העלה דרך SCP
```

**אפשרות 3: SCP מהמחשב המקומי**
```bash
# מהמחשב המקומי (PowerShell)
scp -i your-key.pem -r . ubuntu@YOUR_INSTANCE_IP:~/signlanguage/
```

### 3.4 התקן תלויות

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

---

## שלב 4: הורדת נתונים

```bash
# הורד מ-S3
aws s3 sync s3://your-bucket-name/data/Data/ ./Data/

# או אם העלית ארכיון:
aws s3 cp s3://your-bucket-name/data/sign_language_data.tar.gz ./
tar -xzf sign_language_data.tar.gz
```

---

## שלב 5: אימון המודל

### 5.1 השתמש ב-screen או tmux (חשוב!)

```bash
# התקן screen אם לא קיים
sudo apt-get update
sudo apt-get install screen -y

# צור session חדש
screen -S training

# עכשיו כל מה שתעשה ימשיך גם אם ההתחברות תתנתק
```

### 5.2 הרץ אימון

```bash
# בתוך screen session
cd ~/signlanguage
source venv/bin/activate

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

### 5.3 ניתוק מ-screen

- לחץ `Ctrl+A` ואז `D` (detach)
- Session ימשיך לרוץ ברקע
- להתחבר מחדש: `screen -r training`

### 5.4 שמירת מודל ל-S3 (תכופות!)

```bash
# בתוך screen, או ב-cron job
# שמור checkpoints תכופים
aws s3 sync models/ s3://your-bucket-name/models/ --exclude "*" --include "*.keras"
```

---

## שלב 6: הורדת המודל

```bash
# מהמחשב המקומי
aws s3 sync s3://your-bucket-name/models/ ./models/
```

---

## טיפים חשובים

### 1. Spot Instance Best Practices

```bash
# בדוק אם instance עומד להיפסק
# AWS שולח warning 2 דקות לפני
# אפשר לבדוק ב-console או דרך API

# פתרון: שמור checkpoints תכופים
# המודל שומר אוטומטית את הטוב ביותר
```

### 2. ניטור עלויות

- **CloudWatch**: עקוב אחר שימוש
- **AWS Cost Explorer**: בדוק עלויות
- **תזכורות**: הגדר billing alerts

### 3. אופטימיזציה

```bash
# השתמש ב-mixed precision training (אם נתמך)
# הקטן batch size אם אין מספיק memory
# השתמש ב-GPU memory growth
```

### 4. Backup

```bash
# שמור את המודל ל-S3 כל epoch
# או כל 10 epochs
# אפשר לעשות ב-cron job
```

---

## פקודות שימושיות

```bash
# בדוק GPU usage
watch -n 1 nvidia-smi

# בדוק disk space
df -h

# בדוק memory
free -h

# בדוק CPU
htop

# בדוק אם training רץ
ps aux | grep python

# בדוק logs
tail -f training.log
```

---

## Troubleshooting

### GPU לא מזוהה
```bash
# בדוק drivers
nvidia-smi

# אם לא עובד, התקן:
sudo apt-get update
sudo apt-get install -y nvidia-driver-470
sudo reboot
```

### Out of Memory
```bash
# הקטן batch size
--batch-size 16

# או השתמש ב-gradient accumulation
```

### Connection נקטע
```bash
# השתמש ב-screen או tmux
# או השתמש ב-nohup
nohup python train_model.py ... > training.log 2>&1 &
```

---

## סיום והפסקת Instance

**חשוב מאוד - כבה את ה-instance כדי לא לשלם!**

```bash
# דרך Console:
# EC2 → Instances → Select → Instance State → Stop/Terminate

# דרך CLI:
aws ec2 stop-instances --instance-ids i-xxxxxxxxxxxxx

# או terminate (מוחק את ה-instance):
aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxxx
```

---

## סיכום עלויות

| Instance Type | On-Demand/שעה | Spot/שעה | אימון 4 שעות |
|--------------|---------------|----------|--------------|
| g4dn.xlarge   | $0.50         | ~$0.05   | $0.20-$2.00  |
| g4dn.2xlarge  | $0.75         | ~$0.08   | $0.32-$3.00  |
| g5.xlarge     | $1.00         | ~$0.10   | $0.40-$4.00  |

**עם Spot Instances, אימון יכול לעלות פחות מדולר!**

---

## Next Steps

1. הפעל instance
2. העתק/הורד את הפרויקט
3. הרץ אימון ב-screen
4. שמור מודל ל-S3
5. כבה instance כשסיימת

**זכור: תמיד כבה את ה-instance כשסיימת כדי לא לשלם!**

