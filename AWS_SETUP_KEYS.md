# איך להשיג AWS Access Keys - שלב אחר שלב

## אם אתה משתמש ב-Root Account:

### שלב 1: היכנס ל-AWS Console

1. פתח https://console.aws.amazon.com
2. התחבר עם root account

### שלב 2: פתח Security Credentials

1. **לחץ על השם שלך** (ימין למעלה)
2. **לחץ "Security credentials"**

### שלב 3: צור Access Key

1. **גלול למטה** ל-"Access keys (access key ID and secret access key)"
2. **לחץ "Create access key"**
3. **בחר "Command Line Interface (CLI)"**
4. **סמן את התיבה "I understand..."**
5. **לחץ "Next"**
6. **לחץ "Create access key"**

### שלב 4: שמור את ה-Keys

**חשוב מאוד!** תראה את ה-Keys רק פעם אחת!

1. **Access Key ID**: משהו כמו `AKIAIOSFODNN7EXAMPLE`
2. **Secret Access Key**: משהו כמו `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`

**שמור אותם במקום בטוח!**

---

## אם אתה משתמש ב-IAM User:

### שלב 1: היכנס ל-AWS Console

1. פתח https://console.aws.amazon.com
2. התחבר

### שלב 2: פתח IAM

1. **חפש "IAM"** בשורת החיפוש
2. **לחץ על "IAM"**

### שלב 3: בחר User

1. **בסרגל השמאלי → "Users"**
2. **בחר את ה-user שלך** (או צור חדש)

### שלב 4: צור Access Key

1. **לחץ על הטאב "Security credentials"**
2. **גלול למטה → "Access keys"**
3. **לחץ "Create access key"**
4. **בחר "Command Line Interface (CLI)"**
5. **לחץ "Next"**
6. **לחץ "Create access key"**

### שלב 5: שמור את ה-Keys

**חשוב מאוד!** תראה את ה-Keys רק פעם אחת!

1. **Access Key ID**: העתק
2. **Secret Access Key**: העתק

**שמור אותם במקום בטוח!**

---

## הגדרת AWS CLI עם ה-Keys

```bash
aws configure
```

**מלא:**
- **AWS Access Key ID**: מה-Console
- **AWS Secret Access Key**: מה-Console
- **Default region name**: `us-east-1` (או region אחר)
- **Default output format**: `json`

**עכשיו זה יעבוד!**

---

## בדיקה

```bash
aws s3 ls
```

**אם זה עובד** - תראה רשימה של S3 buckets (או רשימה ריקה אם אין לך buckets).

**אם יש שגיאה** - בדוק שה-Keys נכונים.

---

**בהצלחה!**

