from pathlib import Path
import csv

# Check real files vs CSV
real_files = {f.name for f in Path('Data/Keypoints/rawVideos/GoodBye').glob('*.npy')}
print("Real files:")
for f in sorted(real_files):
    print(f"  {f}")

print("\nCSV entries:")
with open('Data/Labels/dataset.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    csv_files = set()
    for row in reader:
        if 'GOODBYE' in row['path']:
            csv_name = Path(row['path']).name
            csv_files.add(csv_name)
            print(f"  {csv_name}")

print(f"\nMatch: {real_files == csv_files}")
if real_files != csv_files:
    print(f"\nMissing from CSV: {real_files - csv_files}")
    print(f"Extra in CSV: {csv_files - real_files}")

