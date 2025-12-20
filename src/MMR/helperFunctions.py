import datetime
import pandas as pd
import csv
import numpy as np
import os

# ==========================
# UTILITY FUNCTIONS
# ==========================
def generate_run_id():
    # generate unique run ID based on the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{timestamp}"
    return run_id


# ==========================
# MATRIX ALIGNMENT FUNCTIONS
# ==========================

#Extract a submatrix of a DataFrame containing only the specified users and items
def align_matrix_to_user_items(matrix_df, filtered_item_ids, filtered_user_ids):    
    # Find row indices of users in the filtered list that exist in the DataFrame
    user_indices = [matrix_df.index.get_loc(u) for u in filtered_user_ids if u in matrix_df.index]

    # Find column indices of items in the filtered list that exist in the DataFrame
    item_indices = [matrix_df.columns.get_loc(i) for i in filtered_item_ids if i in matrix_df.columns]

    # Extract the submatrix for these users and items as a NumPy array
    aligned_matrix = matrix_df.values[np.ix_(user_indices, item_indices)]

    # Extract the same submatrix as a DataFrame for easier reference
    aligned_df = matrix_df.iloc[user_indices, item_indices]

    return aligned_matrix, aligned_df


#  Aligns training and validation DataFrames to have the same items 
# filters out users with no interactions
def prepare_train_val_matrices(train_df, val_df):
    # Keep only items that exist in both train and validation sets
    common_items = train_df.columns.intersection(val_df.columns)
    train_aligned = train_df[common_items]
    val_aligned = val_df[common_items]

    # Convert training DataFrame to numpy array and filter out users with no ratings
    R_train = train_aligned.values
    user_filter = R_train.sum(axis=1) > 0
    R_filtered_train = R_train[user_filter, :]

    # Store filtered user and item IDs
    filtered_user_ids = train_aligned.index[user_filter].tolist()
    filtered_item_ids = train_aligned.columns.tolist()

    # Align validation matrix to match filtered users and items
    R_filtered_val, val_data_filtered = align_matrix_to_user_items(
        val_aligned,
        filtered_item_ids,
        filtered_user_ids
    )

    # Log shapes for debugging
    print(f"Train matrix: {R_filtered_train.shape}, Val matrix: {R_filtered_val.shape}")

    return (
        R_filtered_train, 
        R_filtered_val,  
        val_data_filtered, 
        filtered_user_ids, 
        filtered_item_ids)



# ==========================
# Extract predicted ratings from a trained MF model for specific filtered users
#==========================
def get_filtered_predictions(trained_mf_model, test_user_ids, train_filtered_user_ids):     
    #  Map user IDs to MF row indice
    user_id_to_idx = {uid: idx for idx, uid in enumerate(train_filtered_user_ids)}
    print(f"[DEBUG] Total training users: {len(train_filtered_user_ids)}")
    print(f"[DEBUG] Sample train_user_to_idx mapping (first 5): {list(user_id_to_idx.items())[:5]}")

    # Check that all test users exist in training
    # missing_users = [uid for uid in test_user_ids if uid not in user_id_to_idx]
    # if missing_users:
    #     print(f"[WARNING] Some test users not in MF model: {missing_users}")
    # else:
    #     print(f"[DEBUG] All test users exist in MF model ({len(test_user_ids)} users)")

    # Get row indices of valid test users
    test_user_indices = [user_id_to_idx[uid] for uid in test_user_ids]
    print(f"[DEBUG] Sample test_user_indices (first 10): {test_user_indices[:10]}")

    # Get full predicted ratings from MF
    predicted_ratings_all = trained_mf_model.full_prediction()
    print(f"[DEBUG] Shape of full predicted ratings: {predicted_ratings_all.shape}")
    
    # Extract predicted ratings only for test users
    predicted_ratings = predicted_ratings_all[test_user_indices, :]
    print(f"[DEBUG] Shape of predicted ratings for test users: {predicted_ratings.shape}")
    print(f"[DEBUG] Sample predicted ratings for first test user:\n{predicted_ratings[0, :10]}")

    return predicted_ratings




# ==========================
# CANDIDATE LIST / MMR INPUT FUNCTIONS
# ==========================
def build_mmr_input(
    candidate_list_csv,
    R_filtered,
    filtered_user_ids,
    filtered_item_ids,
):

    # Load candidate recommendations CSV and keep only rows for filtered users
    df = pd.read_csv(candidate_list_csv)
    df = df[df["userId"].isin(filtered_user_ids)]

    # Ensure item IDs are strings for consistency
    df["itemId"] = df["itemId"].astype(str)

    # Build a list of unique candidate items from the CSV
    candidate_items = df["itemId"].drop_duplicates().tolist()
    if not candidate_items:
        raise ValueError("No candidate items after filtering! Check your item IDs.")

    # Initialize predicted ratings matrix 
    num_items = len(candidate_items)
    num_users = len(filtered_user_ids)
    predicted_ratings_top_n = np.zeros((num_users, num_items))

    # Map user IDs and item IDs to row/column indices in the matrix
    user_to_row = {u: i for i, u in enumerate(filtered_user_ids)}
    item_to_col = {i: j for j, i in enumerate(candidate_items)}

    # Fill matrix with predicted rating from the CSV
    for _, row in df.iterrows():
        user_id, item_id, rating = row["userId"], row["itemId"], row["predictedRating"]
        if user_id in user_to_row and item_id in item_to_col:
            predicted_ratings_top_n[user_to_row[user_id], item_to_col[item_id]] = rating

    # Remove items that have zero prediction for all users 
    non_zero_cols = np.any(predicted_ratings_top_n > 0, axis=0)
    candidate_items = [item for keep, item in zip(non_zero_cols, candidate_items) if keep]
    predicted_ratings_top_n = predicted_ratings_top_n[:, non_zero_cols]

    # Build a mask of items each user has already rated 
    user_history_top_n = []
    for user_idx in range(num_users):
        # Get indices of items the user has rated in the original R_filtered matrix
        rated_indices = np.where(R_filtered[user_idx] > 0)[0]
        rated_item_ids = {str(filtered_item_ids[i]) for i in rated_indices if i < len(filtered_item_ids)}

        # Build a boolean mask for candidate items
        mask = np.zeros(num_items, dtype=bool)
        for j, item_id in enumerate(candidate_items):
            if item_id in rated_item_ids:
                mask[j] = True
        user_history_top_n.append(mask)

    return predicted_ratings_top_n, user_history_top_n, candidate_items


def build_mmr_input_from_nn(
    candidate_list_csv,
    interactions_df=None, 
):

    # Load candidate list from NN output CSV
    df = pd.read_csv(candidate_list_csv)
    df["userId"] = df["userId"].astype(str)
    df["itemId"] = df["itemId"].astype(str)

    # Extract unique users and candidate items
    user_ids = df["userId"].unique().tolist()
    candidate_items = df["itemId"].unique().tolist()

    # Map users/items to row/column indices in the predicted rating matrix
    user_to_row = {u: i for i, u in enumerate(user_ids)}
    item_to_col = {i: j for j, i in enumerate(candidate_items)}

    num_users = len(user_ids)
    num_items = len(candidate_items)

    # Initialize predicted ratings matrix (users x candidate items)
    predicted_ratings = np.zeros((num_users, num_items))

    # Fill matrix with predicted ratings from NN CSV
    for _, row in df.iterrows():
        predicted_ratings[
            user_to_row[row["userId"]],
            item_to_col[row["itemId"]],
        ] = row["rating"]

    # build user history mask
    user_history = None

    if interactions_df is not None:
        interactions_df["userId"] = interactions_df["userId"].astype(str)
        interactions_df["itemId"] = interactions_df["itemId"].astype(str)

        user_history = []

        # For each user, mark items they've already interacted with
        for u in user_ids:
            # Select all items interacted with by the current user u
            seen_items = set(
                interactions_df.loc[
                    interactions_df["userId"] == u, "itemId"
                ]
            )

            # Create a boolean array marking which candidate items the user has already seen
            mask = np.array(
                [item in seen_items for item in candidate_items],
                dtype=bool,
            )
            user_history.append(mask)

    return predicted_ratings, user_history, user_ids, candidate_items


# ==========================
# LOGGING FUNCTIONS
# ==========================
def log_experiment(output_dir, file_name, params):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir,file_name)

    #Cheeck if file exists
    file_exists = os.path.isfile(log_file)

    # Open the CSV file in append mode and write experiment parameters
    with open(log_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=params.keys())
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        # Append the current experiment parameters as a new row
        writer.writerow(params)

    print(f"Logged experiment to {log_file}")


def log_loss_history(output_dir, filename, train_mse, val_mse):
    loss_file = os.path.join(output_dir, filename)

    # Make sure run_id directory exists
    os.makedirs(os.path.dirname(loss_file), exist_ok=True)

    file_exists = os.path.isfile(loss_file)

    with open(loss_file, "a", newline="") as f:
        writer = csv.writer(f)

        # Header only once
        if not file_exists:
            writer.writerow(["Epoch", "Train_mse", "Val_mse"])

        # Write one row per epoch with training and validation MSE
        for epoch, (t, v) in enumerate(zip(train_mse, val_mse)):
            writer.writerow([epoch, float(t), float(v)])


    print(f"Logged experiment to {loss_file}")
