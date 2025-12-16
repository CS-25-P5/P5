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
def align_matrix_to_user_items(matrix_df, filtered_item_ids, filtered_user_ids):    
    # Get indices of users/items that exist in matrix
    user_indices = [matrix_df.index.get_loc(u) for u in filtered_user_ids if u in matrix_df.index]
    item_indices = [matrix_df.columns.get_loc(i) for i in filtered_item_ids if i in matrix_df.columns]

    aligned_matrix = matrix_df.values[np.ix_(user_indices, item_indices)]
    aligned_df = matrix_df.iloc[user_indices, item_indices]

    return aligned_matrix, aligned_df


def align_matrix_to_user(matrix_df, filtered_user_ids):
    user_indices = [
        matrix_df.index.get_loc(u)
        for u in filtered_user_ids
        if u in matrix_df.index
    ]

    aligned_df = matrix_df.iloc[user_indices, :]
    return aligned_df.values, aligned_df


def prepare_train_val_matrices(train_df, val_df):

    # Align train and val to common users/items
    # common_users = train_df.index.intersection(val_df.index)
    # common_items = train_df.columns.intersection(val_df.columns)

    # train_aligned = train_df.loc[common_users, common_items]
    # val_aligned = val_df.loc[common_users, common_items]

    #filter out users with no training interactions
    common_items = train_df.columns.intersection(val_df.columns)
    train_aligned = train_df[common_items]
    val_aligned = val_df[common_items]

    # Convert to numpy and remove users with no interactions
    R_train = train_aligned.values
    user_filter = R_train.sum(axis=1) > 0
    R_filtered_train = R_train[user_filter, :]
    filtered_user_ids = train_aligned.index[user_filter].tolist()
    filtered_item_ids = train_aligned.columns.tolist()
    #filtered_item_titles = [id_to_title[i] for i in filtered_item_ids]

    R_filtered_val, val_data_filtered = align_matrix_to_user_items(
        val_aligned,
        filtered_item_ids,
        filtered_user_ids
    )

    # Log shapes for debugging
    print(f"Train matrix: {R_filtered_train.shape}, Val matrix: {R_filtered_val.shape}")

    return R_filtered_train, R_filtered_val,  val_data_filtered, filtered_user_ids, filtered_item_ids



# ==========================
# PREDICTION FUNCTIONS
#==========================
def get_filtered_predictions(trained_mf_model, filtered_df, train_filtered_user_ids, filtered_item_ids=None):     
    # Get the filtered user and item IDs from the aligned DataFrame
    filtered_user_ids = filtered_df.index.tolist()


    #filtered_item_ids = filtered_df.columns.tolist()

    #print(f"Filtered users: {len(filtered_user_ids)}, Filtered items: {len(filtered_item_ids)}")
    
    # Align filtered items to MF model
    trained_items = np.array([str(i) for i in trained_mf_model.item_ids])
    filtered_item_ids_str = np.array([str(i) for i in filtered_item_ids])

    item_mask = np.isin(trained_items, filtered_item_ids_str)
    item_indices_in_mf = np.where(item_mask)[0]
    
    # Get the predicted ratings for the filtered items
    predicted_ratings_all = trained_mf_model.full_prediction()[:, item_indices_in_mf]
    

    # Map training user IDs to MF model indices
    mf_user_to_idx = {} 
    for idx, user_id in enumerate(train_filtered_user_ids):
        user_str = str(user_id)   # convert user ID to string
        mf_user_to_idx[user_str] = idx
    
    # Get indices of test users in MF predictions
    test_user_indices = []
    for user_id in filtered_user_ids:
        user_str = str(user_id)
        if user_str in mf_user_to_idx:
            test_user_indices.append(mf_user_to_idx[user_str])
        else:
            #test_user_indices.append(0)  #
            raise ValueError(f"User {user_id} not in MF model")
    
    # Extract only predictions for test users
    predicted_ratings = predicted_ratings_all[test_user_indices, :]

    return filtered_user_ids, filtered_item_ids, predicted_ratings




# ==========================
# CANDIDATE LIST / MMR INPUT FUNCTIONS
# ==========================
# def build_mmr_input(
#     candidate_list_csv,
#     R_filtered,
#     filtered_user_ids,
#     filtered_item_ids,
# ):
#     df = pd.read_csv(candidate_list_csv)
#     df = df[df["userId"].isin(filtered_user_ids)]

#     df["itemId"] = df["itemId"].astype(str)
#     filtered_item_ids = list(map(str, filtered_item_ids))
#     df = df[df["itemId"].isin(filtered_item_ids)]

#     candidate_items = []
#     seen = set()
#     for _, row in df.iterrows():
#         if row["itemId"] not in seen:
#             candidate_items.append(row["itemId"])
#             seen.add(row["itemId"])


#     print(f"[DEBUG] Candidate items after filtering: {len(candidate_items)}")

#     num_items = len(candidate_items)
#     predicted_ratings_top_n = np.zeros((len(filtered_user_ids), num_items))

#     user_to_row = {u: i for i, u in enumerate(filtered_user_ids)}
#     item_to_col = {i: j for j, i in enumerate(candidate_items)}

#     for _, row in df.iterrows():
#         if row["userId"] in user_to_row and row["itemId"] in item_to_col:
#             predicted_ratings_top_n[
#                 user_to_row[row["userId"]],
#                 item_to_col[row["itemId"]]
#             ] = row["predictedRating"]

#     user_history_top_n = []

#     for user_idx in range(len(filtered_user_ids)):
#         rated_item_indices = np.where(R_filtered[user_idx] > 0)[0]
#         rated_item_ids = {filtered_item_ids[i] for i in rated_item_indices}

#         mask = np.zeros(num_items, dtype=bool)
#         for j, item_id in enumerate(candidate_items):
#             if item_id in rated_item_ids:
#                 mask[j] = True

#         user_history_top_n.append(mask)

#     return predicted_ratings_top_n, user_history_top_n, candidate_items



def build_mmr_input(
    candidate_list_csv,
    R_filtered,
    filtered_user_ids,
    filtered_item_ids,
):

    # --- Load and filter CSV ---
    df = pd.read_csv(candidate_list_csv)
    df = df[df["userId"].isin(filtered_user_ids)]

    # Ensure type consistency for filtering
    df["itemId"] = df["itemId"].astype(str)
    # filtered_item_ids_str = [str(i) for i in filtered_item_ids]

    # # Filter items that are in filtered_item_ids
    # df = df[df["itemId"].isin(filtered_item_ids_str)]

    # --- Build candidate items list from CSV only ---
    candidate_items = df["itemId"].drop_duplicates().tolist()
    if not candidate_items:
        raise ValueError("No candidate items after filtering! Check your item IDs.")

    # --- Initialize predicted ratings matrix ---
    num_items = len(candidate_items)
    num_users = len(filtered_user_ids)
    predicted_ratings_top_n = np.zeros((num_users, num_items))

    user_to_row = {u: i for i, u in enumerate(filtered_user_ids)}
    item_to_col = {i: j for j, i in enumerate(candidate_items)}

    # Fill matrix with predicted rating
    for _, row in df.iterrows():
        user_id, item_id, rating = row["userId"], row["itemId"], row["predictedRating"]
        if user_id in user_to_row and item_id in item_to_col:
            predicted_ratings_top_n[user_to_row[user_id], item_to_col[item_id]] = rating

    # --- Remove items that have zero prediction for all users ---
    non_zero_cols = np.any(predicted_ratings_top_n > 0, axis=0)
    candidate_items = [item for keep, item in zip(non_zero_cols, candidate_items) if keep]
    predicted_ratings_top_n = predicted_ratings_top_n[:, non_zero_cols]

    # --- Build user history masks ---
    user_history_top_n = []
    for user_idx in range(num_users):
        rated_indices = np.where(R_filtered[user_idx] > 0)[0]
        rated_item_ids = {str(filtered_item_ids[i]) for i in rated_indices if i < len(filtered_item_ids)}

        mask = np.zeros(num_items, dtype=bool)
        for j, item_id in enumerate(candidate_items):
            if item_id in rated_item_ids:
                mask[j] = True
        user_history_top_n.append(mask)

    return predicted_ratings_top_n, user_history_top_n, candidate_items

# ==========================
# LOGGING FUNCTIONS
# ==========================
def log_experiment(output_dir, file_name, params):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir,file_name)

    #Cheeck if file exists
    file_exists = os.path.isfile(log_file)

    #Write to Csv
    with open(log_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=params.keys())
        if not file_exists:
            writer.writeheader()
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

        # One row per epoch
        for epoch, (t, v) in enumerate(zip(train_mse, val_mse)):
            writer.writerow([epoch, float(t), float(v)])


    print(f"Logged experiment to {loss_file}")
