import pandas as pd
import os
from pathlib import Path


def clean_csv_comments(input_folder, encoding='latin1'):
    """
    Scans a folder for CSV files, removes rows starting with '#',
    and saves them in a 'Cleaned_CSVs' subfolder.
    """
    # 1. Setup paths
    input_path = Path(input_folder)
    output_path = input_path / "Cleaned_CSVs"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory: {output_path}")

    # 2. Identify CSV files
    files_to_process = [f for f in os.listdir(input_path) if f.lower().endswith('.csv')]

    if not files_to_process:
        print(f"No CSV files found in {input_path}")
        return

    print(f"Found {len(files_to_process)} CSV files. Starting cleaning...")

    for filename in files_to_process:
        file_path = input_path / filename
        print(f"Processing: {filename}...")

        try:
            # 3. Read CSV
            # Using comment='#' tells pandas to ignore lines starting with '#' automatically
            # We use encoding='latin1' as seen in your previous error logs
            df = pd.read_csv(file_path, encoding=encoding, comment='#')

            # 4. Save to new folder
            save_path = output_path / filename

            # index=False prevents pandas from adding an extra '0, 1, 2' column
            df.to_csv(save_path, index=False, encoding=encoding)
            print(f"  ✅ Saved {len(df)} rows to: {save_path}")

        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")

    print("\nProcess Complete.")


if __name__ == "__main__":
    # Change this path to the folder containing your GoodBooks CSVs
    TARGET_FOLDER = r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)"



    if os.path.exists(TARGET_FOLDER):
        clean_csv_comments(TARGET_FOLDER)
    else:
        print(f"Folder not found: {TARGET_FOLDER}")