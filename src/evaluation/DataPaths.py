import pandas as pd

# Configuration
THRESHOLD = 4.0
K = 10
CALCULATE_ILD = True  # Set to False to skip ILD calculation

# Catalog paths
#CATALOG_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\GoodBooks\books.csv"
CATALOG_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"
#test
CATALOG_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\movies_test2.csv"

CATALOG = pd.read_csv(CATALOG_PATH)
CATALOG = CATALOG.rename(columns={"itemId": "item_id"})


# Item features path (for ILD calculation)
ITEM_FEATURES_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"
#ITEM_FEATURES_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\GoodBooks\books.csv"

ITEM_FEATURES_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\movies_test2.csv"

#Test1
#GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\ratings_test_titles2.csv"

#test2
GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\grount_truth_test2.csv"


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

#GROUND_TRUTH = (r"E:\Data\Diana-NN\Output_Predictions_val_100K_movies(MLPwithBPR)\VAL_GROUNDTRUTH.csv")

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
    #100k movieLens
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run01_Movies.csv", "random Movies 1"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run02_Movies.csv", "random Movies 2"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run03_Movies.csv", "random Movies 3"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run04_Movies.csv", "random Movies 4"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run05_Movies.csv", "random Movies 5"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run06_Movies.csv", "random Movies 6"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run07_Movies.csv", "random Movies 7"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run08_Movies.csv", "random Movies 8"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run09_Movies.csv", "random Movies 9"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run10_Movies.csv", "random Movies 10"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run11_Movies.csv", "random Movies 11"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run12_Movies.csv", "random Movies 12"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run13_Movies.csv", "random Movies 13"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run14_Movies.csv", "random Movies 14"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run15_Movies.csv", "random Movies 15"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run16_Movies.csv", "random Movies 16"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run17_Movies.csv", "random Movies 17"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run18_Movies.csv", "random Movies 18"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run19_Movies.csv", "random Movies 19"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run20_Movies.csv", "random Movies 20"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run21_Movies.csv", "random Movies 21"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run22_Movies.csv", "random Movies 22"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run23_Movies.csv", "random Movies 23"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run24_Movies.csv", "random Movies 24"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run25_Movies.csv", "random Movies 25"),

    #100K goodbooks
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run01_books.csv", "Books 1"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run02_books.csv", "random Books 2"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run03_books.csv", "random Books 3"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run04_books.csv", "random Books 4"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run05_books.csv", "random Books 5"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run06_books.csv", "random Books 6"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run07_books.csv", "random Books 7"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run08_books.csv", "random Books 8"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run09_books.csv", "random Books 9"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run10_books.csv", "random Books 10"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run11_books.csv", "random Books 11"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run12_books.csv", "random Books 12"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run13_books.csv", "random Books 13"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run14_books.csv", "random Books 14"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run15_books.csv", "random Books 15"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run16_books.csv", "random Books 16"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run17_books.csv", "random Books 17"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run18_books.csv", "random Books 18"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run19_books.csv", "random Books 19"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run20_books.csv", "random Books 20"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run21_books.csv", "random Books 21"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run22_books.csv", "random Books 22"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run23_books.csv", "random Books 23"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run24_books.csv", "random Books 24"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\random_top10_run25_books.csv", "random Books 25"),

    #test2
   (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\test2_predictions.csv", "Test2"),

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

    # # #DPP - movies
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\2025-12-17_00-49-51\dpp_test_100000_jaccard_top_10.csv", "dpp_Jaccard"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\2025-12-17_00-49-51\dpp_test_100000_cosine_top_10.csv", "dpp_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\2025-12-17_00-49-51\mf_test_100000_top_10.csv", "MF"),

    # DPP - books
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\books\2025-12-16_21-42-31\dpp_test_100000_cosine_top_10.csv", "dpp_Jaccard"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\books\2025-12-16_21-42-31\dpp_test_100000_jaccard_top_10.csv", "dpp_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\books\2025-12-16_21-42-31\mf_test_100000_top_10.csv", "MF"),


##### Johannes #######

    # #NN johannes - movies ml100k
    # #1layer
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # #2 layers
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    #
    # #3 layers
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml100k\predictions\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

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

    # # #diana val total data 100K movies with BPR new
    # # # #1 LAYER
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed32_lr0001_batch64_filtered.csv" ,"One-32-0001-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed32_lr0001_batch128_filtered.csv" ,"One-32-0001-b128"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed32_lr00003_batch64_filtered.csv" ,"One-32-00003-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed32_lr00003_batch128_filtered.csv" ,"One-32-00003-b128"),
    #
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommend_BPRnn_OneLayer_embed64_lr0001_batch64_filtered.csv","One-64-0001-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed64_lr0001_batch128_filtered.csv","One-64-0001-128"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed64_lr00003_batch64_filtered.csv","One-64-00003-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed64_lr00003_batch128_filtered.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed32_lr0001_batch64_filtered.csv","Two-32-0001-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed32_lr0001_batch128_filtered.csv","Two-32-0001-b128"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed32_lr00003_batch64_filtered.csv","Two-32-00003-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed32_lr00003_batch128_filtered.csv","Two-32-00003-b128"),
    #
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed64_lr0001_batch64_filtered.csv","Two-64-0001-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed64_lr0001_batch128_filtered.csv","Two-64-0001-b128"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed64_lr00003_batch64_filtered.csv","Two-64-00003-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed64_lr00003_batch128_filtered.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed32_lr0001_batch64_filtered.csv","Three-32-0001-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed32_lr0001_batch128_filtered.csv","Three-32-0001-b128"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed32_lr00003_batch64_filtered.csv","Three-32-00003-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed32_lr00003_batch128_filtered.csv","Three-32-00003-b128"),
    #
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed64_lr0001_batch64_filtered.csv","Three-64-0001-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed64_lr0001_batch128_filtered.csv","Three-64-0001-b128"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed64_lr00003_batch64_filtered.csv","Three-64-00003-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed64_lr00003_batch128_filtered.csv","Three-64-00003-b128"),


    # # # #diana total data 100K goodbooks with BPR new
    # # # # #1 LAYER
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64_filtered.csv",
    #     "One-32-0001-b64"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128_filtered.csv",
    #     "One-32-0001-b128"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64_filtered.csv",
    #     "One-32-00003-b64"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128_filtered.csv",
    #     "One-32-00003-b128"),
    #
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64_filtered.csv",
    #     "One-64-0001-b64"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128_filtered.csv",
    #     "One-64-0001-128"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64_filtered.csv",
    #     "One-64-00003-b64"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128_filtered.csv",
    #     "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64_filtered.csv",
    #     "Two-32-0001-b64"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128_filtered.csv",
    #     "Two-32-0001-b128"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64_filtered.csv",
    #     "Two-32-00003-b64"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128_filtered.csv",
    #     "Two-32-00003-b128"),
    #
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64_filtered.csv",
    #     "Two-64-0001-b64"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128_filtered.csv",
    #     "Two-64-0001-b128"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64_filtered.csv",
    #     "Two-64-00003-b64"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128_filtered.csv",
    #     "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64_filtered.csv",
    #     "Three-32-0001-b64"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128_filtered.csv",
    #     "Three-32-0001-b128"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64_filtered.csv",
    #     "Three-32-00003-b64"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128_filtered.csv",
    #     "Three-32-00003-b128"),
    #
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64_filtered.csv",
    #     "Three-64-0001-b64"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128_filtered.csv",
    #     "Three-64-0001-b128"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64_filtered.csv",
    #     "Three-64-00003-b64"),
    # (
    #     r"E:\Data\Diana-NN\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128_filtered.csv",
    #     "Three-64-00003-b128"),


    # # #diana total data 1M movies with BPR new
    # # # # #1 LAYER
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64_filtered.csv",
    # "One-32-0001-b64"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128_filtered.csv",
    # "One-32-0001-b128"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64_filtered.csv",
    # "One-32-00003-b64"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128_filtered.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64_filtered.csv",
    # "One-64-0001-b64"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128_filtered.csv",
    # "One-64-0001-128"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64_filtered.csv",
    # "One-64-00003-b64"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128_filtered.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64_filtered.csv",
    # "Two-32-0001-b64"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128_filtered.csv",
    # "Two-32-0001-b128"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64_filtered.csv",
    # "Two-32-00003-b64"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128_filtered.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64_filtered.csv",
    # "Two-64-0001-b64"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128_filtered.csv",
    # "Two-64-0001-b128"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64_filtered.csv",
    # "Two-64-00003-b64"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128_filtered.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64_filtered.csv",
    # "Three-32-0001-b64"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128_filtered.csv",
    # "Three-32-0001-b128"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64_filtered.csv",
    # "Three-32-00003-b64"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128_filtered.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64_filtered.csv",
    # "Three-64-0001-b64"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128_filtered.csv",
    # "Three-64-0001-b128"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64_filtered.csv",
    # "Three-64-00003-b64"),
    # (
    # r"E:\Data\Diana-NN\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128_filtered.csv",
    # "Three-64-00003-b128"),

    # # #diana total data 100K movies with BPR new
    # # # #1 LAYER
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64_filtered.csv" ,"One-32-0001-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128_filtered.csv" ,"One-32-0001-b128"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64_filtered.csv" ,"One-32-00003-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128_filtered.csv" ,"One-32-00003-b128"),
    #
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64_filtered.csv","One-64-0001-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128_filtered.csv","One-64-0001-128"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64_filtered.csv","One-64-00003-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128_filtered.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64_filtered.csv","Two-32-0001-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128_filtered.csv","Two-32-0001-b128"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64_filtered.csv","Two-32-00003-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128_filtered.csv","Two-32-00003-b128"),
    #
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64_filtered.csv","Two-64-0001-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128_filtered.csv","Two-64-0001-b128"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64_filtered.csv","Two-64-00003-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128_filtered.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64_filtered.csv","Three-32-0001-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128_filtered.csv","Three-32-0001-b128"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64_filtered.csv","Three-32-00003-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128_filtered.csv","Three-32-00003-b128"),
    #
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64_filtered.csv","Three-64-0001-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128_filtered.csv","Three-64-0001-b128"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64_filtered.csv","Three-64-00003-b64"),
    # (r"E:\Data\Diana-NN\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128_filtered.csv","Three-64-00003-b128"),

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

