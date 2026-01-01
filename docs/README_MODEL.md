# Sign Language Recognition - GRU Model

מדריך לאימון ושימוש במודל GRU לזיהוי שפת סימנים.

## מבנה הפרויקט

```
signlanguage/
├── Data/
│   ├── Keypoints/          # Keypoints extracted from videos (.npy files)
│   ├── Labels/             # CSV files with dataset splits
│   └── ...
├── models/                 # Trained models (created after training)
├── data_loader.py          # Data loading utilities
├── model_gru.py            # GRU model architecture
├── train_model.py          # Training script
├── predict.py              # Inference script
├── extract_keypoints.py    # Extract keypoints from videos
├── create_dataset_csv.py   # Create CSV dataset files
└── aws_setup.py            # AWS setup utilities
```

## התקנת תלויות

```bash
pip install -r requirements.txt
```

## יצירת Dataset CSV

אם עדיין לא יצרת CSV:
```bash
python create_dataset_csv.py
```

זה ייצור `Data/Labels/dataset.csv` עם חלוקה ל-train/val/test.

## אימון המודל

### אימון מקומי (אם יש לך GPU)

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

### אימון ב-AWS

ראה `AWS_TRAINING_GUIDE.md` לפרטים מלאים.

**קיצור דרך - EC2:**
1. הפעל EC2 instance עם GPU (g4dn.xlarge או g5.xlarge)
2. העתק את הפרויקט:
   ```bash
   scp -r -i your-key.pem . ubuntu@YOUR_INSTANCE_IP:~/signlanguage/
   ```
3. התחבר והרץ:
   ```bash
   ssh -i your-key.pem ubuntu@YOUR_INSTANCE_IP
   cd signlanguage
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python train_model.py --csv Data/Labels/dataset.csv ...
   ```

## שימוש במודל (Prediction)

### חיזוי מסרטון

```bash
python predict.py \
    --model models/run_YYYYMMDD_HHMMSS/best_model.keras \
    --video path/to/video.mp4
```

### חיזוי מקבצי keypoints

```bash
python predict.py \
    --model models/run_YYYYMMDD_HHMMSS/best_model.keras \
    --keypoints Data/Keypoints/rawVideos/Hello/Hello01.npy
```

### שמירת תוצאות ל-JSON

```bash
python predict.py \
    --model models/run_YYYYMMDD_HHMMSS/best_model.keras \
    --video path/to/video.mp4 \
    --output results.json
```

## מבנה המודל

המודל כולל:
- **2-3 שכבות GRU** עם dropout
- **שכבות Dense** עם activation ReLU
- **Output layer** עם softmax למיון

**Input:** Sequences of keypoints (num_frames, 126 features)
- 126 features = 2 hands × 21 keypoints × 3 coordinates (x, y, z)
- Keypoints מנורמלים (wrist at origin, scaled by hand size)

**Output:** Probability distribution over sign language classes

## פרמטרים של האימון

- `--batch-size`: גודל batch (default: 32)
- `--epochs`: מספר מקסימלי של epochs (default: 100)
- `--gru-units`: מספר יחידות בכל שכבות GRU (default: 128)
- `--num-gru-layers`: מספר שכבות GRU (default: 2)
- `--dropout`: Dropout rate (default: 0.3)
- `--learning-rate`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 10)

## Callbacks

האימון משתמש ב:
- **ModelCheckpoint**: שומר את המודל הטוב ביותר לפי validation accuracy
- **EarlyStopping**: עוצר אם אין שיפור ב-patience epochs
- **ReduceLROnPlateau**: מקטין learning rate אם validation loss לא משתפר

## קבצי Output

לאחר האימון, בתיקיית `models/run_TIMESTAMP/`:
- `best_model.keras`: המודל הטוב ביותר (לפי validation accuracy)
- `final_model.keras`: המודל מהאימון האחרון
- `model_architecture.json`: ארכיטקטורת המודל
- `label_mapping.json`: מיפוי labels
- `training_history.json`: היסטוריית האימון
- `test_results.json`: תוצאות על test set
- `training_params.json`: פרמטרי האימון

## שיפור המודל

אם תוצאות האימון לא מספיק טובות:
1. **הגדל את מספר ה-epochs**
2. **נסה מספר גדול יותר של GRU units** (256, 512)
3. **הוסף שכבות GRU נוספות**
4. **שנה dropout rate**
5. **הגדל את גודל ה-dataset** - אוסף יותר סרטונים
6. **נסה Data Augmentation** - הוסף variations לנתונים

## הערות

- המודל דורש קבצי keypoints מנורמלים (כמו ב-`extract_keypoints.py`)
- Sequence length מוגדר לפי ה-max length ב-training set
- המודל משתמש ב-sparse categorical crossentropy loss (למיון)

## Troubleshooting

**Out of Memory:**
- הקטן `batch_size`
- הקטן `gru_units`

**Overfitting:**
- הגדל `dropout_rate`
- השתמש ב-regularization נוסף

**Underfitting:**
- הגדל מספר layers או units
- הגדל מספר epochs
- בדוק את איכות הנתונים

