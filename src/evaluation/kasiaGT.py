import pandas as pd
import os


def extract_ground_truth(input_path, output_path=None):
    """
    Extracts ground truth ratings from a CSV file and reformats it.

    Input columns:  userId, itemId, true_rating, predictedRating
    Output columns: user_id, itemId, rating

    Args:
        input_path: Path to input CSV file
        output_path: Optional output path. If None, appends '_gt' to original filename
    """
    print(f"Reading data from: {input_path}")

    # Load the CSV file
    df = pd.read_csv(input_path)

    print(f"Original columns: {list(df.columns)}")
    print(f"Original rows: {len(df)}")

    # Validate required columns exist
    required_cols = ['userId', 'itemId', 'true_rating']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create new dataframe with desired format
    output_df = pd.DataFrame()
    output_df['user_id'] = df['userId']
    output_df['itemId'] = df['itemId']
    output_df['rating'] = df['true_rating']

    # Generate output path if not provided
    if output_path is None:
        directory = os.path.dirname(input_path)
        base_name = os.path.basename(input_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(directory, f"{name_without_ext}_gt.csv")

    # Save the result
    output_df.to_csv(output_path, index=False)

    print(f"\n✅ Extracted ground truth successfully!")
    print(f"   Output columns: {list(output_df.columns)}")
    print(f"   Output rows: {len(output_df)}")
    print(f"   Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    # === CONFIGURE THIS PATH ===
    INPUT_FILE = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\mf_test_100000_predictions.csv"

    # Optional: specify custom output path
    # OUTPUT_FILE = r"C:\path\to\output\ground_truth.csv"
    OUTPUT_FILE = None  # Will auto-generate filename

    extract_ground_truth(INPUT_FILE, OUTPUT_FILE)