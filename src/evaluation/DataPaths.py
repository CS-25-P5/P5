import pandas as pd

# Configuration
THRESHOLD = 4.0
K = 5
CALCULATE_ILD = True  # Set to False to skip ILD calculation

# Catalog paths
#CATALOG_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\GoodBooks\books.csv"
CATALOG_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"

CATALOG = pd.read_csv(CATALOG_PATH)
CATALOG = CATALOG.rename(columns={"itemId": "item_id"})

# Item features path (for ILD calculation)
ITEM_FEATURES_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"
#ITEM_FEATURES_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\GoodBooks\books.csv"

#Test1
#GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\ratings_test_titles2.csv"

#test2
#GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\grount_truth_test2.csv"


#MF/MMR - li movies
#
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies_groundtruth.csv")

#MF - li books
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books_groundtruth.csv")

#NN - diana

#movies 100k - Diana
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\TEST_GROUNDTRUTH\ratings_100K_movies_test.csv")

#movies 1m - Diana
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\TEST_GROUNDTRUTH\ratings_1M_movies_test.csv")

#books 100k - Diana
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\TEST_GROUNDTRUTH\ratings_100K_goodbooks_test.csv")


#NN - johannes
#movies 100k
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml100k\predictions\ground_truth")
#Books
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\gb100k\predictions\ground_truth")
#movies 1m
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_resultater\ml1m\predictions\ground_truth")


#DPP - movies
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\mf_test_100000_predictions_gt.csv")

#DPP - books
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\books\books_ratings_100000_test_gt.csv")

#validation ground truth

#Books 100k
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\VAL_GROUNDTRUTH.csv")

#ml 100k
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\VAL_GROUNDTRUTH.csv")

#1m
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\VAL_GROUNDTRUTH.csv")

# Models to compare
MODELS = [

    ##### Test #####

    #random recommendations

    #test1
    #(r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\mmr_data\test_predictions.csv", "Test"),

    #test2
   # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\test2_predictions.csv", "Test2"),

 ##### Li #######

    #MMR - li movies 100k R=1 R=0.5
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies-2025-12-16_05-59-25-(R=1)\mf_test_100000_top_50.csv", "R=1,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies-2025-12-16_05-59-25-(R=1)\mmr_test_100000_cosine_top_50.csv", "R=1,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies-2025-12-16_05-59-25-(R=1)\mmr_test_100000_jaccard_top_50.csv", "R=1,MMR_jaccard"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies - 2025-12-16_06-01-48-(R=0.5)\mf_test_100000_top_50.csv", "R=0.5,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies - 2025-12-16_06-01-48-(R=0.5)\mmr_test_100000_cosine_top_50.csv", "R=0.5,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies - 2025-12-16_06-01-48-(R=0.5)\mmr_test_100000_jaccard_top_50.csv", "R=0.5,MMR_jaccard"),

    #MMR - li books - 100k R=1, R=0.5, R=0.6
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_06-11-03-(R=1)\mf_test_100000_top_50.csv", "R=1,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_06-11-03-(R=1)\mmr_test_100000_cosine_top_50.csv", "R=1,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_06-11-03-(R=1)\mmr_test_100000_jaccard_top_50.csv", "R=1,MMR_jaccard"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_06-13-36-(R=0.5)\mf_test_100000_top_50.csv", "R=0.5,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_06-13-36-(R=0.5)\mmr_test_100000_cosine_top_50.csv", "R=0.5,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_06-13-36-(R=0.5)\mmr_test_100000_jaccard_top_50.csv", "R=0.5,MMR_jaccard"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_06-12-55-(R=0.6)\mf_test_100000_top_50.csv", "R=0.6,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_06-12-55-(R=0.6)\mmr_test_100000_cosine_top_50.csv", "R=0.6,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_06-12-55-(R=0.6)\mmr_test_100000_jaccard_top_50.csv", "R=0.6,MMR_jaccard"),

    # MMR - li books 100k R=1, R=0.6
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_16-26-52-(R=1)\mf_test_100000_top_50.csv", "Books, R=1,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_16-26-52-(R=1)\mmr_test_100000_cosine_top_50.csv", "Books,R=0.5,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_16-26-52-(R=1)\mmr_test_100000_jaccard_top_50.csv", "Books, R=0.5,MMR_jaccard"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_16-35-32-(R=0.6)\mf_test_100000_top_50.csv", "Books, R=0.6,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_16-35-32-(R=0.6)\mmr_test_100000_cosine_top_50.csv", "Books, R=0.6,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books - 2025-12-16_16-35-32-(R=0.6)\mmr_test_100000_jaccard_top_50.csv", "Books, R=0.6,MMR_jaccard"),

    # # MMR - li movies - 100k R=1, R=0.5
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies-2025-12-16_16-13-38-(R=1.0)\mf_test_100000_top_50.csv", "Movies, R=1,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies-2025-12-16_16-13-38-(R=1.0)\mmr_test_100000_cosine_top_50.csv", "Movies, R=1,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies-2025-12-16_16-13-38-(R=1.0)\mmr_test_100000_jaccard_top_50.csv", "Movies, R=1,MMR_jaccard"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies-2025-12-16_16-18-45-(R=0.5)\mf_test_100000_top_50.csv", "Movies, R=0.5,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies-2025-12-16_16-18-45-(R=0.5)\mmr_test_100000_cosine_top_50.csv", "Movies, R=0.5,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies-2025-12-16_16-18-45-(R=0.5)\mmr_test_100000_jaccard_top_50.csv", "Movies, R=0.5,MMR_jaccard"),
    #

    ###### Kasia #######

    # #DPP - movies
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\dpp_test_100000_jaccard_top_10_with_genres.csv", "dpp_Jaccard"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\dpp_test_100000_cosine_top_10_with_genres.csv", "dpp_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\mf_test_100000_top_10_with_genres.csv", "MF"),

    # DPP - books
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\books\dpp_test_100000_jaccard_top_10.csv", "dpp_Jaccard"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\books\dpp_test_100000_cosine_top_10.csv", "dpp_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\books\mf_test_100000_top_10.csv", "MF"),


##### Johannes #######

    # #NN johannes - movies ml100k
    # #1layer
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),

    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),

    #2 layers
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),

    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),


    #3 layers
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

    # NN johannes - books
    # 1layer
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv", "1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # #2 layers
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # #3 layers
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\gb100k\predictions\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

    # NN johannes - movies 1m
    # # 1layer
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv", "1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # #2 layers
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # #3 layers
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),

#### Diana #####

    # # #diana val data 100K movies with BPR
    # # # #1 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr0001_batch64.csv" ,"One-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr0001_batch128.csv" ,"One-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr00003_batch64.csv" ,"One-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr00003_batch128.csv" ,"One-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr0001_batch64.csv","One-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr0001_batch128.csv","One-64-0001-128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr00003_batch64.csv","One-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr00003_batch128.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr0001_batch64.csv","Two-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr0001_batch128.csv","Two-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr00003_batch64.csv","Two-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr00003_batch128.csv","Two-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr0001_batch64.csv","Two-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr0001_batch128.csv","Two-64-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr00003_batch64.csv","Two-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr00003_batch128.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr0001_batch64.csv","Three-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr0001_batch128.csv","Three-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr00003_batch64.csv","Three-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr00003_batch128.csv","Three-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr0001_batch64.csv","Three-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr0001_batch128.csv","Three-64-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr00003_batch64.csv","Three-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr00003_batch128.csv","Three-64-00003-b128"),

    # #diana validation data 1M movies with BPR
    # 1 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr0001_batch64.csv" ,"One-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr0001_batch128.csv" ,"One-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr00003_batch64.csv" ,"One-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr00003_batch128.csv" ,"One-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr0001_batch64.csv","One-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr0001_batch128.csv","One-64-0001-128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr00003_batch64.csv","One-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr00003_batch128.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr0001_batch64.csv","Two-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr0001_batch128.csv","Two-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr00003_batch64.csv","Two-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr00003_batch128.csv","Two-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr0001_batch64.csv","Two-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr0001_batch128.csv","Two-64-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr00003_batch64.csv","Two-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr00003_batch128.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr0001_batch64.csv","Three-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr0001_batch128.csv","Three-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr00003_batch64.csv","Three-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr00003_batch128.csv","Three-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr0001_batch64.csv","Three-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr0001_batch128.csv","Three-64-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr00003_batch64.csv","Three-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr00003_batch128.csv","Three-64-00003-b128"),

    # # #diana validation data 100K books with BPR
    # # #1 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed32_lr0001_batch64.csv" ,"One-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed32_lr0001_batch128.csv" ,"One-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed32_lr00003_batch64.csv" ,"One-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed32_lr00003_batch128.csv" ,"One-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed64_lr0001_batch64.csv","One-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed64_lr0001_batch128.csv","One-64-0001-128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed64_lr00003_batch64.csv","One-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed64_lr00003_batch128.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr0001_batch64.csv","Two-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr0001_batch128.csv","Two-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr00003_batch64.csv","Two-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr00003_batch128.csv","Two-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr0001_batch64.csv","Two-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr0001_batch128.csv","Two-64-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr00003_batch64.csv","Two-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr00003_batch128.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr0001_batch64.csv","Three-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr0001_batch128.csv","Three-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr00003_batch64.csv","Three-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr00003_batch128.csv","Three-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr0001_batch64.csv","Three-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr0001_batch128.csv","Three-64-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr00003_batch64.csv","Three-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr00003_batch128.csv","Three-64-00003-b128"),

    # diana validation data 100K movies with Genres
    # # #1 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),

    # #diana validation data 1M movies with Genres
    # 1 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),

    # #diana validation data 100K books with Genres
    # 1 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),

    # # #diana data 100K movies with BPR
    # # # #1 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64.csv" ,"One-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128.csv" ,"One-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64.csv" ,"One-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128.csv" ,"One-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv","One-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128.csv","One-64-0001-128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64.csv","One-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64.csv","Two-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128.csv","Two-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64.csv","Two-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128.csv","Two-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64.csv","Two-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128.csv","Two-64-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64.csv","Two-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64.csv","Three-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128.csv","Three-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64.csv","Three-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128.csv","Three-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64.csv","Three-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128.csv","Three-64-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64.csv","Three-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128.csv","Three-64-00003-b128"),

    # #diana data 100K movies with Genres
    # #1 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),

    # # #diana data 100K movies with BPR Total
    # # #1 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64.csv" ,"One-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128.csv" ,"One-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64.csv" ,"One-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128.csv" ,"One-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv","One-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128.csv","One-64-0001-128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64.csv","One-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64.csv","Two-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128.csv","Two-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64.csv","Two-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128.csv","Two-32-00003-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128.csv","Two-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64.csv","Two-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128.csv","Two-64-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64.csv","Two-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64.csv","Three-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128.csv","Three-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64.csv","Three-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128.csv","Three-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64.csv","Three-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128.csv","Three-64-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64.csv","Three-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128.csv","Three-64-00003-b128"),

    # # #diana data 1M movies with BPR
    # #1 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64.csv" ,"One-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128.csv" ,"One-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64.csv" ,"One-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128.csv" ,"One-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv","One-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128.csv","One-64-0001-128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64.csv","One-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64.csv","Two-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128.csv","Two-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64.csv","Two-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128.csv","Two-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64.csv","Two-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128.csv","Two-64-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64.csv","Two-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64.csv","Three-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128.csv","Three-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64.csv","Three-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128.csv","Three-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64.csv","Three-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128.csv","Three-64-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64.csv","Three-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128.csv","Three-64-00003-b128"),

    # #diana data 1M movies with Genres
    # 1 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),

    # # # #diana data 1m movies with BPR Total
    # # # #1 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_1M_movies_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),

    # # #diana data 100K books with BPR
    # # #1 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64.csv" ,"One-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128.csv" ,"One-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64.csv" ,"One-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128.csv" ,"One-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv","One-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128.csv","One-64-0001-128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64.csv","One-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64.csv","Two-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128.csv","Two-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64.csv","Two-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128.csv","Two-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64.csv","Two-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128.csv","Two-64-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64.csv","Two-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64.csv","Three-32-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128.csv","Three-32-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64.csv","Three-32-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128.csv","Three-32-00003-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64.csv","Three-64-0001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128.csv","Three-64-0001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64.csv","Three-64-00003-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128.csv","Three-64-00003-b128"),

    # #diana data 100K books with Genres
    # 1 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),

    # # #diana data 100K books with BPR Total
    # # #1 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\dianas_resulter\Recommend_test_100K_goodbooks_Total(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),

]

