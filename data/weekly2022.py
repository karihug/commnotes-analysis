import subprocess
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

start_date = datetime(2022, 6, 1)
end_date = datetime(2023, 6, 1)

# Use absolute paths here (change these accordingly if needed)
base_dir = Path("/scratch/users/commnotes/communitynotes2022/communitynotes/static/sourcecode")
data_dir = Path("/scratch/users/commnotes/")
original_ratings_path = base_dir / "merged_ratings2023.tsv"
filtered_ratings_path = base_dir / "ratings_filtered.tsv"
notes_path = data_dir/ "communitynotes/sourcecode/notes-00000.tsv"
enrollment_path = data_dir/ "communitynotes/sourcecode/userEnrollment-00000.tsv"
note_status_history_path = base_dir/"noteStatusHistory-00000.tsv"
main_script_absolute = base_dir / "main.py"

output_root = base_dir / "weekly_outputs"
output_root.mkdir(exist_ok=True)

print("Loading full ratings dataset...")
ratings = pd.read_csv(original_ratings_path, sep='\t', dtype={'createdAtMillis': 'float'})

current_date = start_date

while current_date <= end_date:
    cutoff_timestamp_ms = current_date.timestamp() * 1000
    ratings_filtered = ratings[ratings['createdAtMillis'] < cutoff_timestamp_ms]
    ratings_filtered.to_csv(filtered_ratings_path, sep='\t', index=False)

    date_str = current_date.strftime("%Y-%m-%d")
    weekly_output_dir = output_root / f"output_{date_str}"
    weekly_output_dir.mkdir(exist_ok=True)

    cmd = [
        "python",
        str(main_script_absolute),
        "--enrollment", str(enrollment_path),
        "--notes_path", str(notes_path),
        "--ratings_path", str(filtered_ratings_path)
,        "--note_status_history_path", str(note_status_history_path)  # <-- Add this line explicitly
    ]    

    print(f"\nRunning scoring for cutoff date: {date_str}")

    subprocess.run(cmd, cwd=weekly_output_dir, check=True)

    current_date += timedelta(days=7)

print("🎉 All weekly runs completed.")

