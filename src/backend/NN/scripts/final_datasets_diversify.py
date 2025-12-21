
import os
import pandas

def prep_files(input_folder):
    

    os.makedirs(input_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):

        input_path=os.path.join(input_folder, filename)
        dataset_df = pandas.read_csv(input_path) 

        drop_rating = dataset_df.drop(columns=["rating"])
        rename_column = drop_rating.rename(columns={"recommendation_score":"rating"})


        #keep only 50 highest per user 
        finals = (rename_column.groupby("userId").head(50))


        #Create output filename 
        base_path, extension = os.path.splitext(filename)
        output_filename = f"{base_path}_final{extension}"
        output_path = os.path.join(input_folder, output_filename)

        #Save
        finals.to_csv(output_path, index = False)

prep_files("data/OUTPUT_datasets/NN/LI")