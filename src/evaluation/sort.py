import pandas as pd
import os


def sort_csv_by_user(input_path, output_path=None):
    """
    Sorts a CSV file by userId column.

    Args:
        input_path: Path to input CSV file
        output_path: Optional output path. If None, appends '_sorted' to filename
    """
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    initial_rows = len(df)
    print(f"Loaded {initial_rows} rows")

    # Sort by userId
    df_sorted = df.sort_values('userId')

    # Reset index to keep it clean
    df_sorted = df_sorted.reset_index(drop=True)

    # Generate output path if not provided
    if output_path is None:
        directory = os.path.dirname(input_path)
        base_name = os.path.basename(input_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(directory, f"{name_without_ext}_sorted.csv")

    # Save sorted file
    df_sorted.to_csv(output_path, index=False)

    print(f"✅ Sorted and saved to: {output_path}")
    print(f"   First few userIds: {df_sorted['userId'].head().tolist()}")
    print(f"   Last few userIds: {df_sorted['userId'].tail().tolist()}")

    return output_path


if __name__ == "__main__":
    # === CONFIGURE THIS PATH ===
    INPUT_FILE =r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\TEST_GROUNDTRUTH\ratings_100K_movies_test.csv"

    # Optional: specify custom output path
    # OUTPUT_FILE = r"C:\path\to\sorted_file.csv"
    OUTPUT_FILE = None  # Will auto-generate filename

    sort_csv_by_user(INPUT_FILE, OUTPUT_FILE)