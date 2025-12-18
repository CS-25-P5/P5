import os
import pandas

def choose_best_config_BPR(recom_directory, val_path, 
                           output_dir, k):
    
    
    os.makedirs(output_dir, exist_ok=True)

    val_df=pandas.read_csv(val_path)
    validation_positives = val_df[["user_id", "itemId"]].drop_duplicates() #Interaction in validation set. DO NOT HAVE TRAINING WITH!
    evaluation_users = set(validation_positives["user_id"].unique())

    recom_directory_path = os.listdir(recom_directory)
    output_dir_path = os.listdir(output_dir)

    for files in recom_directory_path:
        if not files.endswith("_filtered.csv"):
            continue
        recom_file_path = os.path.join(recom_directory, files)
        recommendation_file = pandas.read_csv(recom_file_path)
        #We only check evaluation users and keep all seen and unseen interactions 
        recommendation_file = recommendation_file[recommendation_file["user_id"].isin(evaluation_users)]

        recommendation_file=recommendation_file.merge(
            validation_positives.assign(label=1),           #if it has label 1 it means it is in the validation set!
            on = ["user_id", "itemId"],
            how = "left"
        )

        #Otherwise label user-items that are nto in val as negatives
        recommendation_file["label"] = recommendation_file["label"].fillna(0).astype(int)

        #Sort top 30
        recommendation_file = recommendation_file.sort_values(["user_id", "recommendation_score"], ascending=[True, False])
        topk = recommendation_file.groupby("user_id", as_index=False).head(k)

        output_file = topk[["user_id", "itemId", "rating", "recommendation_score", "label"]]

        name, extention = os.path.split(files)
        output_path = os.path.join(output_dir, f"{name}{"_val_eval"}{extention}")
        output_file.to_csv(output_path, index=False)


'''
choose_best_config_BPR(recom_directory = "data/OUTPUT_datasets/NN/Recommend_test_100K_movies_TOTAL(MLPwithBPR)",
                        val_path = "data/INPUT_VAL/ratings_100K_movies_val.csv", 
                        output_dir = "data/OUTPUT_datasets/NN/Recommend_test_100K_movies_TOTAL(MLPwithBPR)", 
                        k = 20)

choose_best_config_BPR(recom_directory = "data/OUTPUT_datasets/NN/Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)",
                        val_path = "data/INPUT_VAL/ratings_100K_goodbooks_val.csv", 
                        output_dir = "data/OUTPUT_datasets/NN/Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)", 
                        k = 20)
'''


