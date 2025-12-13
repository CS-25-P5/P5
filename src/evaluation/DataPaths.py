import pandas as pd

# Configuration
THRESHOLD = 4.0  # for the metrics that need to view things in a binary fashion
K = 10  # recommendations to look at

# SET THIS TO False TO SKIP ILD AND GENRE LOADING
CALCULATE_ILD = True # Change to False to skip ILD entirely

#CATALOG_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"
CATALOG_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\GoodBooks\books.csv"
CATALOG = pd.read_csv(CATALOG_PATH)
CATALOG = CATALOG.rename(columns={"itemId": "item_id"})

#Test1
#GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\ratings_test_titles2.csv"

#test2
#GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\grount_truth_test2.csv"


#MF - li movies
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movies_ratings_100000_test.csv")
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movie\movies_ratings_100000_test.csv")

#MF - li books
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\book\books_ratings_100000_train.csv")

#NN - diana

#movies 100k
GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\TEST_GROUNDTRUTH\ratings_100K_movies_test.csv")

#movies 1m
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\TEST_GROUNDTRUTH\ratings_1M_movies_test.csv")

#books 100k
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\TEST_GROUNDTRUTH\ratings_100K_goodbooks_test.csv")


#NN - johannes
#movies 100k
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\ground_truth")

#Books
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\ground_truth")

#movies 1m
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\ground_truth")


#DPP - movies
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\2025-12-10_20-10-18\gt_ratings.csv")

#DPP - books
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\books\books_ratings_100000_test_gt.csv")

# Models to compare
MODELS = [
    #test1
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\test_predictions.csv", "Test"),

    #test2
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\test2_predictions.csv", "Test2"),

    #mf
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\movie\ALIGNED_mf_test_predictions.csv", "mf"),

    #MMR - li movies
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movie\mf_test_100000_predictions.csv", "MF"),
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movie\mmr_test_100000_cosine_predictions.csv", "MMR_cosine"),
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movie\mmr_test_100000_jaccard_predictions.csv", "MMR_jaccard"),

    #MMR - li books
    #r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\book\mf_test_100000_predictions.csv", "MF"),
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\book\mmr_test_100000_cosine_predictions.csv", "MMR_cosine"),
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\book\mmr_test_100000_jaccard_predictions.csv", "MMR_jaccard"),

    #DPP - movies
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\2025-12-10_20-10-18\dpp_test_100000_jaccard_top_10.csv", "dpp_Jaccard"),
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\2025-12-10_20-10-18\dpp_test_100000_cosine_top_10.csv", "dpp_cosine"),
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\2025-12-10_20-10-18\mf_test_100000_predictions.csv", "MF"),

    # DPP - books
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\books\ALIGNED_dpp_train_jaccard_recommendations.csv","dpp_jaccard"),
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\books\ALIGNED_dpp_train_cosine_recommendations.csv","dpp_cosine"),
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\books\ALIGNED_mf_train_predictions_books.csv", "MF"),

    # #diana data 100K movies with BPR
    # #1 LAYER
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64.csv" ,"One-32-0001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128.csv" ,"One-32-0001-b128"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64.csv" ,"One-32-00003-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128.csv" ,"One-32-00003-b128"),

    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv","One-64-0001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128.csv","One-64-0001-128"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64.csv","One-64-00003-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128.csv","One-64-00003-b128"),

    #2 LAYER
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64.csv","Two-32-0001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128.csv","Two-32-0001-b128"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64.csv","Two-32-00003-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128.csv","Two-32-00003-b128"),

    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64.csv","Two-64-0001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128.csv","Two-64-0001-b128"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64.csv","Two-64-00003-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128.csv","Two-64-00003-b128"),

    # 3 LAYER
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64.csv","Three-32-0001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128.csv","Three-32-0001-b128"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64.csv","Three-32-00003-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128.csv","Three-32-00003-b128"),

    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64.csv","Three-64-0001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128.csv","Three-64-0001-b128"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64.csv","Three-64-00003-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128.csv","Three-64-00003-b128"),

    # diana data 100K movies with Genres
    # 1 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_TwoLayer_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_TwoLayer_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_TwoLayer_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_TwoLayer_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_TwoLayer_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_TwoLayer_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_TwoLayer_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_TwoLayer_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_ThreeLayer_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_ThreeLayer_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_ThreeLayer_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_ThreeLayer_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_ThreeLayer_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_ThreeLayer_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_ThreeLayer_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\RecommendBPRnn_ThreeLayer_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),

    # diana data 100K movies with BPR TOtAL
    # 1 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),


    # diana data 1M movies with BPR Total
    # 1 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayer_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayer_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),
    #

    #NN johannes - movies
    #1layer
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # #2 layers
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    #
    # #3 layers
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

    # NN johannes - books
    # 1layer
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv", "1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # #2 layers
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # #3 layers
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

    # NN johannes - movies 1m
    # # 1layer
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv", "1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # #2 layers
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # #3 layers
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
]