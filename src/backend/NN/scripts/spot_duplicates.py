import pandas
import os
'''
final_dataset= pandas.read_csv("data/INPUT_datasets/ratingsandgenres_100K_movies_DUPE.csv")

final_dataset[final_dataset.duplicated(keep=False)].head(10)

dups = final_dataset[
    final_dataset.duplicated(subset=["userId", "movieId"], keep=False)
]

print(dups.sort_values(["userId", "movieId"]).head(10))
'''


def remove_training_from_recommendation(recommendation_dir_path, training_file_path, output_dir_path):
    # Output directory
    os.makedirs(output_dir_path, exist_ok=True)
    train = pandas.read_csv(training_file_path)
    

    train["user_id"] = pandas.to_numeric(train["user_id"], errors = "raise").astype(int)
    train["itemId"] = pandas.to_numeric(train["itemId"], errors = "raise").astype(int)
    
    useritems_fromtrain = train[["user_id", "itemId"]].drop_duplicates()
    


    #LOOP THRU the FILES
    for myfile in os.listdir(recommendation_dir_path):

        if not myfile.endswith(".csv"):
            continue
        recommendation_file_path = os.path.join(recommendation_dir_path, myfile)
        recommendations = pandas.read_csv(recommendation_file_path)

        recommendations["user_id"] = pandas.to_numeric(recommendations["user_id"], errors = "raise").astype(int)
        recommendations["itemId"] = pandas.to_numeric(recommendations["itemId"], errors = "raise").astype(int)



        filter_me_out = recommendations.merge(useritems_fromtrain, on=["user_id", "itemId"], how="left", 
                                              indicator=True).query('_merge == "left_only"').drop(columns="_merge")

        name, extension = os.path.splitext(myfile)
        new_filename = name + "_filtered" + extension
        output_file_path = os.path.join(output_dir_path, new_filename)

        filter_me_out.to_csv(output_file_path, index=False)




'''
remove_training_from_recommendation(recommendation_dir_path = "data/OUTPUT_datasets/NN/Recommend_test_100K_movies_TOTAL(MLPwithBPR)",
                                    training_file_path = "data/INPUT_TRAIN/ratings_100K_movies_train.csv",
                                    output_dir_path = "data/OUTPUT_datasets/NN/Recommend_test_100K_movies_TOTAL(MLPwithBPR)")

'''

'''
remove_training_from_recommendation(recommendation_dir_path = "data/OUTPUT_datasets/NN/Recommend_test_1M_movies_TOTAL(MLPwithBPR)",
                                    training_file_path = "data/INPUT_TRAIN/ratings_1M_movies_train.csv",
                                    output_dir_path = "data/OUTPUT_datasets/NN/Recommend_test_1M_movies_TOTAL(MLPwithBPR)")
'''

'''
remove_training_from_recommendation(recommendation_dir_path = "data/OUTPUT_datasets/NN/Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)",
                                    training_file_path = "data/INPUT_TRAIN/ratings_100K_goodbooks_train.csv",
                                    output_dir_path = "data/OUTPUT_datasets/NN/Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)")
'''


def reorderfiltered(input_folder):

    for filename in os.listdir(input_folder):
        if filename.endswith("_filtered.csv"):
            input_path = os.path.join(input_folder, filename) #### GEt the the path for each file


            #Load me
            pred_dataset = pandas.read_csv(input_path, comment = "#") 

            #Sorty by userId first, then descend

            sort_by_id = pred_dataset.sort_values(
                by = ["userId", "recommendation_score"],
                ascending=[True, False]
            )

            #Create output filename 
            #base_path, extension = os.path.splitext(filename)
            #output_filename = f"{base_path}_ranked{extension}"
            #output_path = os.path.join(input_folder, output_filename)

            #Save
            sort_by_id.to_csv(input_path, index = False)

            
reorderfiltered("data/OUTPUT_datasets/NN/Recommend_test_1M_movies_TOTAL(MLPwithBPR)")
reorderfiltered("data/OUTPUT_datasets/NN/Recommend_test_100K_movies_TOTAL(MLPwithBPR)")