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

    filtered_user_ids_aligned = [filtered_user_ids[i] for i in user_indices]


    aligned_df = matrix_df.iloc[user_indices, :]
    return aligned_df.values, aligned_df, filtered_user_ids_aligned



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
    filtered_item_ids = filtered_df.columns.tolist()


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
    mf_user_to_idx = {str(u): idx for idx, u in enumerate(train_filtered_user_ids)}



    ## Build predicted_ratings for test users
    predicted_ratings = []
    for user_id in filtered_user_ids:
        if str(user_id) in mf_user_to_idx:
            predicted_ratings.append(predicted_ratings_all[mf_user_to_idx[str(user_id)]])
        else:
            # unseen user → fill with global mean
            predicted_ratings.append(np.full(len(item_indices_in_mf), trained_mf_model.mu))

    predicted_ratings = np.vstack(predicted_ratings)

    return filtered_user_ids, filtered_item_ids, predicted_ratings





# ==========================
# CANDIDATE LIST / MMR INPUT FUNCTIONS
# ==========================
def build_mmr_input(
        candidate_list_csv,
        R_filtered,
        filtered_user_ids,
        filtered_item_ids,
):
    df = pd.read_csv(candidate_list_csv)
    df = df[df["userId"].isin(filtered_user_ids)]
    #df = df[df["itemId"].isin(filtered_item_ids)]

    candidate_items = []
    seen = set()
    for _, row in df.iterrows():
        if row["itemId"] not in seen:
            candidate_items.append(row["itemId"])
            seen.add(row["itemId"])

    num_items = len(candidate_items)
    predicted_ratings_top_n = np.zeros((len(filtered_user_ids), num_items))

    user_to_row = {u: i for i, u in enumerate(filtered_user_ids)}
    item_to_col = {i: j for j, i in enumerate(candidate_items)}

    for _, row in df.iterrows():
        if row["userId"] in user_to_row and row["itemId"] in item_to_col:
            predicted_ratings_top_n[
                user_to_row[row["userId"]],
                item_to_col[row["itemId"]]
            ] = row["predictedRating"]

    user_history_top_n = []

    for user_idx in range(len(filtered_user_ids)):
        rated_item_indices = np.where(R_filtered[user_idx] > 0)[0]
        rated_item_ids = {filtered_item_ids[i] for i in rated_item_indices}

        mask = np.zeros(num_items, dtype=bool)
        for j, item_id in enumerate(candidate_items):
            if item_id in rated_item_ids:
                mask[j] = True

        user_history_top_n.append(mask)

    return predicted_ratings_top_n, user_history_top_n, candidate_items

# def build_mmr_input(candidate_list_csv, R_filtered, filtered_user_ids, filtered_item_ids):
#     # Load candidate CSV and filter for known users/items
#     df = pd.read_csv(candidate_list_csv)
#     df = df[df["userId"].isin(filtered_user_ids) & df["itemId"].isin(filtered_item_ids)]

#     # Build unique candidate items
#     candidate_items = df["itemId"].unique().tolist()
#     num_users = len(filtered_user_ids)
#     num_items = len(candidate_items)

#     # Map user/item IDs to matrix indices
#     user_to_row = {u: i for i, u in enumerate(filtered_user_ids)}
#     item_to_col = {i: j for j, i in enumerate(candidate_items)}

#     # Vectorized relevance matrix
#     user_indices = df["userId"].map(user_to_row).to_numpy(dtype=int)
#     item_indices = df["itemId"].map(item_to_col).to_numpy(dtype=int)
#     predicted_ratings_top_n = np.zeros((num_users, num_items))
#     predicted_ratings_top_n[user_indices, item_indices] = df["predictedRating"].to_numpy()

#     # Vectorized user history mask
#     user_history_top_n = []
#     filtered_item_array = np.array(filtered_item_ids)
#     for user_idx in range(num_users):
#         rated_item_indices = np.where(R_filtered[user_idx] > 0)[0]
#         rated_item_ids = set(filtered_item_array[rated_item_indices])
#         mask = np.isin(candidate_items, list(rated_item_ids))
#         user_history_top_n.append(mask)

#     return predicted_ratings_top_n, user_history_top_n, candidate_items


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