import pandas as pd

# Configuration
THRESHOLD = 4.0
K = 10
CALCULATE_ILD = True  # Set to False to skip ILD calculation

# #movies
# ITEM_FEATURES_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"
# CATALOG_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\MovieLens\movies.csv"

# Books
CATALOG_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\GoodBooks\books.csv"
ITEM_FEATURES_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\GoodBooks\books.csv"

#test2
# CATALOG_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\movies_test2.csv"
# ITEM_FEATURES_PATH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\movies_test2.csv"


CATALOG = pd.read_csv(CATALOG_PATH)
CATALOG = CATALOG.rename(columns={"itemId": "item_id"})


# #Test1
# GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\ratings_test_titles2.csv"

# #test2
# GROUND_TRUTH = r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\grount_truth_test2.csv"


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


#DPP - movies
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\mf_test_100000_predictions_gt.csv")

#DPP - books
#GROUND_TRUTH = (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\books\books_ratings_100000_test_gt.csv")

#DIANA FINAL 18/12
#GROUND_TRUTH = (r"E:\Data\GT\ratings_100K_movies_test.csv")
# GROUND_TRUTH = (r"E:\Data\GT\ratings_1M_movies_test.csv")
#GROUND_TRUTH = (r"E:\Data\GT\ratings_100K_goodbooks_test.csv")

#validation ground truth
#GROUND_TRUTH = (r"E:\Data\GT\ratings_100K_movies_val.csv")
#GROUND_TRUTH = (r"E:\Data\GT\ratings_100K_goodbooks_val.csv")

#johannes gt 12/12
#val
#10ok movies
# GROUND_TRUTH = (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\GT_val.csv")
#1m movies
# GROUND_TRUTH = (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\GT_val.csv")
#100k books
#GROUND_TRUTH = (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\GT_val.csv")


#########
#final
#100 movies
# GROUND_TRUTH = (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\GT_test.csv")
#100k books
#GROUND_TRUTH = (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\gb100k\predictions\GT_test.csv")
# #1m movies
# GROUND_TRUTH = (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\GT_test.csv")
############



#diana val gt

#GROUND_TRUTH = r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\VAL_GROUNDTRUTH.csv"
#GROUND_TRUTH = r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100K_movies(MLPwithGenres)\VAL_GROUNDTRUTH.csv"
# GROUND_TRUTH = r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100K_goodbooks(MLPwithGenres)\VAL_GROUNDTRUTH.csv"

#val pbr
#GROUND_TRUTH = r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\VAL_GROUNDTRUTH.csv"
# GROUND_TRUTH = r"E:\Data\Output_Predictions_val_100K_movies(MLPwithBPR)\VAL_GROUNDTRUTH.csv"
# GROUND_TRUTH = r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\VAL_GROUNDTRUTH.csv"

#diana 10% Test GT
#GROUND_TRUTH = r"E:\Data\INPUT_TEST\ratings_1M_movies_test.csv"
# GROUND_TRUTH = r"E:\Data\INPUT_TEST\ratings_100k_movies_test.csv"
#GROUND_TRUTH = r"E:\Data\INPUT_TEST\ratings_100k_goodbooks_test.csv"

#johannes test:
#book
GROUND_TRUTH = r"E:\Data\dataaaa\gb100k\predictions\ground_truth"
#100k movie
# GROUND_TRUTH = r"E:\Data\dataaaa\ml100k\predictions\ground_truth"
# #1m movie
# GROUND_TRUTH = r"E:\Data\dataaaa\ml1m\predictions\ground_truth"

#1M ground truth
#GROUND_TRUTH = r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_1M_movies(MLPwithGenres)\Cleaned_CSVs\GROUNDTRUTH_alluserandmovies.csv"

# Models to compare
MODELS = [




    ##### Li #######

    # MMR - li movies 100k R=1 R=0.5
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Movies-2025-12-17_14-21-43-(R=0.8)\mf_test_100000_predictions.csv", "R=0.8,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Movies-2025-12-17_14-24-49-(R=0.4)\mf_test_100000_predictions.csv", "R=0.4,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Movies-2025-12-17_14-25-25-(R=0.2)\mf_test_100000_predictions.csv", "R=0.2,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Movies - 2025-12-17_14-21-34-(R=1.0)\mf_test_100000_predictions.csv", "R=1,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Movies - 2025-12-17_14-22-37-(R=0.6)\mf_test_100000_predictions.csv", "R=0.6,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Movies - 2025-12-17_14-23-30-(R=0.5)\mf_test_100000_predictions.csv", "R=0.5,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Movies - 2025-12-17_14-26-42-(R=0.0)\mf_test_100000_predictions.csv", "R=0.0,MF"),

    # MMR - li books - 100k R=1, R=0.5, R=0.6
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Books-12-17_14-32-05-(R=0.2)\mf_test_100000_predictions.csv", "R=0.2,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Books-2025-12-17_14-28-21-(R=0.8)\mf_test_100000_predictions.csv", "R=0.8, MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Books-2025-12-17_14-39-22-(R=0.4)\mf_test_100000_predictions.csv", "R=0.4,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Books -2025-12-17_14-34-25-(R=0.0)\mf_test_100000_predictions.csv", "R=0.0,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Books - 2025-12-17_14-28-13-(R=1.0)\mf_test_100000_predictions.csv", "R=1.0,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Books - 2025-12-17_14-29-17-(R=0.6)\mf_test_100000_predictions.csv", "R=0.6,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Books - 2025-12-17_14-30-21-(R=0.5)\mf_test_100000_predictions.csv", "R=0.5,MF"),
    #

    # MMR - li books 100k R=1, R=0.8 0.6 0.5  0.4 0.0
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-53-18-(R=1.0)\mf_test_100K_top_50.csv", "Books, R=1,MF"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-53-18-(R=1.0)\mmr_test_100k_cosine_top_50.csv", "Books,R=1,MMR_cosine"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-53-18-(R=1.0)\mmr_test_100k_jaccard_top_50.csv", "Books, R=1,MMR_jaccard"),

    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-55-05-(R=0.8)\mf_test_100K_top_50.csv", "Books, R=0.8,MF"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-55-05-(R=0.8)\mmr_test_100k_cosine_top_50.csv", "Books, R=0.8,MMR_cosine"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-55-05-(R=0.8)\mmr_test_100k_jaccard_top_50.csv", "Books, R=0.8,MMR_jaccard"),

    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-55-19-(R=0.6)\mf_test_100K_top_50.csv", "Books, R=0.6, MF"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-55-19-(R=0.6)\mmr_test_100k_cosine_top_50.csv", "Books,R=0.6,MMR_cosine"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-55-19-(R=0.6)\mmr_test_100k_jaccard_top_50.csv", "Books, R=0.6,MMR_jaccard"),

    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-54-00-(R=0.5)\mf_test_100K_top_50.csv", "Books, R=0.5,MF"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-54-00-(R=0.5)\mmr_test_100k_cosine_top_50.csv", "Books, R=0.5,MMR_cosine"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-54-00-(R=0.5)\mmr_test_100k_jaccard_top_50.csv", "Books, R=0.5,MMR_jaccard"),

    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-55-35-(R=0.4)\mf_test_100K_top_50.csv", "Books, R=0.4,MF"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-55-35-(R=0.4)\mmr_test_100k_cosine_top_50.csv", "Books,R=0.4,MMR_cosine"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-55-35-(R=0.4)\mmr_test_100k_jaccard_top_50.csv", "Books, R=0.4,MMR_jaccard"),

    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-55-57-(R=0.0)\mf_test_100K_top_50.csv", "Books, R=0.0,MF"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-55-57-(R=0.0)\mmr_test_100k_cosine_top_50.csv", "Books, R=0.0,MMR_cosine"),
    (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\books\2025-12-19_23-55-57-(R=0.0)\mmr_test_100k_jaccard_top_50.csv", "Books, R=0.0,MMR_jaccard"),


    # # MMR - li movies - 100k R=1, R=0.8 0.6 0.5  0.4 0.0
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-40-49-(R=1.0)\mf_test_100K_top_50.csv", "Books, R=1,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-40-49-(R=1.0)\mmr_test_100k_cosine_top_50.csv", "Books,R=1,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-40-49-(R=1.0)\mmr_test_100k_jaccard_top_50.csv", "Books, R=1,MMR_jaccard"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-40-59-(R=0.8)\mf_test_100K_top_50.csv", "Books, R=0.8,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-40-59-(R=0.8)\mmr_test_100k_cosine_top_50.csv", "Books, R=0.8,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-40-59-(R=0.8)\mmr_test_100k_jaccard_top_50.csv", "Books, R=0.8,MMR_jaccard"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-42-08-(R=0.5)\mf_test_100K_top_50.csv", "Books, R=0.5,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-42-08-(R=0.5)\mmr_test_100k_cosine_top_50.csv", "Books, R=0.5,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-42-08-(R=0.5)\mmr_test_100k_jaccard_top_50.csv", "Books, R=0.5,MMR_jaccard"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-43-04-(R=0.4)\mf_test_100K_top_50.csv", "Books, R=0.4,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-43-04-(R=0.4)\mmr_test_100k_cosine_top_50.csv", "Books,R=0.4,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-43-04-(R=0.4)\mmr_test_100k_jaccard_top_50.csv", "Books, R=0.4,MMR_jaccard"),
    #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-43-24-(R=0.2)\mf_test_100K_top_50.csv", "Books, R=0.2, MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-43-24-(R=0.2)\mmr_test_100k_cosine_top_50.csv", "Books,R=0.2,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-43-24-(R=0.2)\mmr_test_100k_jaccard_top_50.csv", "Books, R=0.2,MMR_jaccard"),
    # #
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-43-50-(R=0.0)\mf_test_100K_top_50.csv", "Books, R=0.0,MF"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-43-50-(R=0.0)\mmr_test_100k_cosine_top_50.csv", "Books, R=0.0,MMR_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\li_resultater\movies\2025-12-19_23-43-50-(R=0.0)\mmr_test_100k_jaccard_top_50.csv", "Books, R=0.0,MMR_jaccard"),

    ##### Test #####

    # #random recommendations
    # 100k movieLens
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
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\evaluation\Random\popularity_top10_20251217_143214.csv", "popularity 25"),

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

    # #test2
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\test\test2_predictions.csv", "Test2"),
    #



##### Johannes #######

    # johannes final runs 100k movies 18/12
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # #2 layers
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # #
    # # #3 layers
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml100k\predictions\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

    # #NN johannes - final run movies ml1M 18/12
    #1layer
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # #2 layers
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    #
    # #3 layers
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\predictions (all user-item pairs)\ml1m\predictions\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

    # #NN johannes - 1m ml final run 18/12
    # #1layer
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_1layers_embed32_lr0.001_batch64.csv", "1layer-em32-lr001-b64"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv", "1layer-em32-lr0003-b64"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv", "1layer-em32-lr0003-b128"),
    #
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_1layers_embed64_lr0.001_batch64.csv", "1layer-em64-lr001-b64"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_1layers_embed64_lr0.001_batch128.csv", "1layer-em64-lr001-b128"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv", "1layer-em64-lr0003-b64"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv", "1layer-em64-lr0003-b128"),
    #
    # # 2 layers
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_2layers_embed32_lr0.001_batch64.csv", "2layer-em32-lr001-b64"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_2layers_embed32_lr0.001_batch128.csv", "2layer-em32-lr001-b128"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv", "2layer-em32-lr0003-b64"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv", "2layer-em32-lr0003-b128"),
    #
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_2layers_embed64_lr0.001_batch64.csv", "2layer-em64-lr001-b64"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_2layers_embed64_lr0.001_batch128.csv", "2layer-em64-lr001-b128"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv", "2layer-em64-lr0003-b64"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv", "2layer-em64-lr0003-b128"),
    #
    # # 3 layers
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_3layers_embed32_lr0.001_batch64.csv", "3layer-em32-lr001-b64"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_3layers_embed32_lr0.001_batch128.csv", "3layer-em32-lr001-b128"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv", "3layer-em32-lr0003-b64"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv", "3layer-em32-lr0003-b128"),
    #
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_3layers_embed64_lr0.001_batch64.csv", "3layer-em64-lr001-b64"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_3layers_embed64_lr0.001_batch128.csv", "3layer-em64-lr001-b128"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em64-lr0003-b64"),
    # (r"E:\Data\dataaaa\ml1m\predictions\MLP_3layers_embed64_lr0.0003_batch128.csv", "3layer-em64-lr0003-b128"),



    # #NN johannes - 100k ml final run 18/12
    # #1layer
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_1layers_embed32_lr0.001_batch64.csv", "1layer-em32-lr001-b64"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv", "1layer-em32-lr0003-b64"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv", "1layer-em32-lr0003-b128"),
    #
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_1layers_embed64_lr0.001_batch64.csv", "1layer-em64-lr001-b64"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_1layers_embed64_lr0.001_batch128.csv", "1layer-em64-lr001-b128"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv", "1layer-em64-lr0003-b64"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv", "1layer-em64-lr0003-b128"),
    #
    # # 2 layers
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_2layers_embed32_lr0.001_batch64.csv", "2layer-em32-lr001-b64"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_2layers_embed32_lr0.001_batch128.csv", "2layer-em32-lr001-b128"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv", "2layer-em32-lr0003-b64"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv", "2layer-em32-lr0003-b128"),
    #
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_2layers_embed64_lr0.001_batch64.csv", "2layer-em64-lr001-b64"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_2layers_embed64_lr0.001_batch128.csv", "2layer-em64-lr001-b128"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv", "2layer-em64-lr0003-b64"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv", "2layer-em64-lr0003-b128"),
    #
    # # 3 layers
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_3layers_embed32_lr0.001_batch64.csv", "3layer-em32-lr001-b64"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_3layers_embed32_lr0.001_batch128.csv", "3layer-em32-lr001-b128"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv", "3layer-em32-lr0003-b64"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv", "3layer-em32-lr0003-b128"),
    #
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_3layers_embed64_lr0.001_batch64.csv", "3layer-em64-lr001-b64"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_3layers_embed64_lr0.001_batch128.csv", "3layer-em64-lr001-b128"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em64-lr0003-b64"),
    # (r"E:\Data\dataaaa\ml100k\predictions\MLP_3layers_embed64_lr0.0003_batch128.csv", "3layer-em64-lr0003-b128"),

    # # # #NN johannes - gb final run 18/12
    # # # #1layer
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr0003-b64"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr0003-b128"),
    #
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em64-lr001-b64"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em64-lr001-b128"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_1layers_embed64_lr0.0003_batch64.csv","1layer-em64-lr0003-b64"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em64-lr0003-b128"),
    #
    # #2 layers
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr0003-b64"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr0003-b128"),
    #
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em64-lr001-b64"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em64-lr001-b128"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em64-lr0003-b64"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em64-lr0003-b128"),
    #
    #
    # #3 layers
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr0003-b64"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr0003-b128"),
    #
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em64-lr001-b64"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em64-lr001-b128"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em64-lr0003-b64"),
    # (r"E:\Data\dataaaa\gb100k\predictions\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em64-lr0003-b128"),



    #johannes validation runs 100k movies 18/12
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_1layers_embed64_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # #2 layers
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # #
    # # #3 layers
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml100k\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

    # #NN johannes - gb validation 18/12
    # #1layer
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed64_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # #2 layers
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    #
    # #3 layers
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

    # #NN johannes - movies ml1M 18/12
    # #1layer
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed64_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # #2 layers
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    #
    # #3 layers
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\gb100k\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),

    # # NN johannes - ml1m final 18/12
    # # 1layer
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_1layers_embed32_lr0.001_batch128.csv", "1layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_1layers_embed32_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_1layers_embed32_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_1layers_embed64_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_1layers_embed64_lr0.001_batch128.csv","1layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_1layers_embed64_lr0.0003_batch64.csv","1layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_1layers_embed64_lr0.0003_batch128.csv","1layer-em32-lr001-b128"),
    #
    # #2 layers
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_2layers_embed32_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_2layers_embed32_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_2layers_embed32_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_2layers_embed32_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_2layers_embed64_lr0.001_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_2layers_embed64_lr0.001_batch128.csv","2layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_2layers_embed64_lr0.0003_batch64.csv","2layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_2layers_embed64_lr0.0003_batch128.csv","2layer-em32-lr001-b128"),
    #
    #
    # #3 layers
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_3layers_embed32_lr0.001_batch64.csv","3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_3layers_embed32_lr0.001_batch128.csv","3layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_3layers_embed32_lr0.0003_batch64.csv","3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_3layers_embed32_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
    #
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_3layers_embed64_lr0.001_batch64.csv",  "3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_3layers_embed64_lr0.001_batch128.csv",  "3layer-em32-lr001-b128"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_3layers_embed64_lr0.0003_batch64.csv", "3layer-em32-lr001-b64"),
    # (r"E:\Data\data for entire set (fixed for real this time)\data for entire set (fixed for real this time)\validation predictions\ml1m\MLP_3layers_embed64_lr0.0003_batch128.csv","3layer-em32-lr001-b128"),
    #

    # NN johannes - movies 1m
    # # 1layer
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\johannes_data_(entire_set)\ml1m\MLP_1layers_embed32_lr0.001_batch64.csv","1layer-em32-lr001-b64"),
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

    # #diana genres movies 100k 18/12 right
    # # #1 LAYER
    # (
    #     r"E:\Data\Recommend_test_100K_movies(MLPwithGenres)\NNgenres_OneLayer_embed32_lr0001_batch64.csv",
    #     "One-32-0001-b64"),
    # (
    #     r"E:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_OneLayer_embed32_lr0001_batch128.csv",
    #     "One-32-0001-b128"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_OneLayer_embed32_lr00003_batch64.csv",
    #     "One-32-00003-b64"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_OneLayer_embed32_lr00003_batch128.csv",
    #     "One-32-00003-b128"),
    #
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_OneLayer_embed64_lr0001_batch64.csv",
    #     "One-64-0001-b64"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_OneLayer_embed64_lr0001_batch128.csv",
    #     "One-64-0001-128"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_OneLayer_embed64_lr00003_batch64.csv",
    #     "One-64-00003-b64"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_OneLayer_embed64_lr00003_batch128.csv",
    #     "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_TwoLayers_embed32_lr0001_batch64.csv",
    #     "Two-32-0001-b64"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_TwoLayers_embed32_lr0001_batch128.csv",
    #     "Two-32-0001-b128"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_TwoLayers_embed32_lr00003_batch64.csv",
    #     "Two-32-00003-b64"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_TwoLayers_embed32_lr00003_batch128.csv",
    #     "Two-32-00003-b128"),
    #
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_TwoLayers_embed64_lr0001_batch64.csv",
    #     "Two-64-0001-b64"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_TwoLayers_embed64_lr0001_batch128.csv",
    #     "Two-64-0001-b128"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_TwoLayers_embed64_lr00003_batch64.csv",
    #     "Two-64-00003-b64"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_TwoLayers_embed64_lr00003_batch128.csv",
    #     "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_ThreeLayers_embed32_lr0001_batch64.csv",
    #     "Three-32-0001-b64"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_ThreeLayers_embed32_lr0001_batch128.csv",
    #     "Three-32-0001-b128"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_ThreeLayers_embed32_lr00003_batch64.csv",
    #     "Three-32-00003-b64"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_ThreeLayers_embed32_lr00003_batch128.csv",
    #     "Three-32-00003-b128"),
    #
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_ThreeLayers_embed64_lr0001_batch64.csv",
    #     "Three-64-0001-b64"),
    # (
    #     r"e:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_ThreeLayers_embed64_lr0001_batch128.csv",
    #     "Three-64-0001-b128"),
    # (
    #     r"E:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_ThreeLayers_embed64_lr00003_batch64.csv",
    #     "Three-64-00003-b64"),
    # (
    #     r"E:\Data\Recommend_test_100K_movies(MLPwithGenres)\nnGenres_ThreeLayers_embed64_lr00003_batch128.csv",
    #     "Three-64-00003-b128"),

    # # #diana bpr movies 1m 18/12
    # # # #1 LAYER
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"E:\Data\Recommend_test_1M_movies(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),

    # # #diana genres movies 1m 18/12
    # # # #1 LAYER
    # (
    #     r"E:\Data\Recommend_test_1M_movies(MLPwithGenres)\NNgenres_OneLayer_embed32_lr0001_batch64.csv",
    #     "One-32-0001-b64"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_OneLayer_embed32_lr0001_batch128.csv",
    #     "One-32-0001-b128"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_OneLayer_embed32_lr00003_batch64.csv",
    #     "One-32-00003-b64"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_OneLayer_embed32_lr00003_batch128.csv",
    #     "One-32-00003-b128"),
    #
    # (
    #     r"E:\Data\Recommend_test_1M_movies(MLPwithGenres)\nngenres_OneLayer_embed64_lr0001_batch64.csv",
    #     "One-64-0001-b64"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_OneLayer_embed64_lr0001_batch128.csv",
    #     "One-64-0001-128"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_OneLayer_embed64_lr00003_batch64.csv",
    #     "One-64-00003-b64"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_OneLayer_embed64_lr00003_batch128.csv",
    #     "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_TwoLayers_embed32_lr0001_batch64.csv",
    #     "Two-32-0001-b64"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_TwoLayers_embed32_lr0001_batch128.csv",
    #     "Two-32-0001-b128"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_TwoLayers_embed32_lr00003_batch64.csv",
    #     "Two-32-00003-b64"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_TwoLayers_embed32_lr00003_batch128.csv",
    #     "Two-32-00003-b128"),
    #
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_TwoLayers_embed64_lr0001_batch64.csv",
    #     "Two-64-0001-b64"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_TwoLayers_embed64_lr0001_batch128.csv",
    #     "Two-64-0001-b128"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_TwoLayers_embed64_lr00003_batch64.csv",
    #     "Two-64-00003-b64"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_TwoLayers_embed64_lr00003_batch128.csv",
    #     "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_ThreeLayers_embed32_lr0001_batch64.csv",
    #     "Three-32-0001-b64"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_ThreeLayers_embed32_lr0001_batch128.csv",
    #     "Three-32-0001-b128"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_ThreeLayers_embed32_lr00003_batch64.csv",
    #     "Three-32-00003-b64"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_ThreeLayers_embed32_lr00003_batch128.csv",
    #     "Three-32-00003-b128"),
    #
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_ThreeLayers_embed64_lr0001_batch64.csv",
    #     "Three-64-0001-b64"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_ThreeLayers_embed64_lr0001_batch128.csv",
    #     "Three-64-0001-b128"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_ThreeLayers_embed64_lr00003_batch64.csv",
    #     "Three-64-00003-b64"),
    # (
    #     r"E:\Data\Recommend_test_1m_movies(MLPwithGenres)\nngenres_ThreeLayers_embed64_lr00003_batch128.csv",
    #     "Three-64-00003-b128"),

    # #diana bpr goodbooks 100k 18/12
    # # #1 LAYER
    #(
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),

    # # # #diana genres goodbooks 100k 18/12
    # # # #1 LAYER
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\NNgenres_OneLayer_embed32_lr0001_batch64.csv",
    #     "One-32-0001-b64"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_OneLayer_embed32_lr0001_batch128.csv",
    #     "One-32-0001-b128"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_OneLayer_embed32_lr00003_batch64.csv",
    #     "One-32-00003-b64"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_OneLayer_embed32_lr00003_batch128.csv",
    #     "One-32-00003-b128"),
    #
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_OneLayer_embed64_lr0001_batch64.csv",
    #     "One-64-0001-b64"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_OneLayer_embed64_lr0001_batch128.csv",
    #     "One-64-0001-128"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_OneLayer_embed64_lr00003_batch64.csv",
    #     "One-64-00003-b64"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_OneLayer_embed64_lr00003_batch128.csv",
    #     "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_TwoLayers_embed32_lr0001_batch64.csv",
    #     "Two-32-0001-b64"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_TwoLayers_embed32_lr0001_batch128.csv",
    #     "Two-32-0001-b128"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_TwoLayers_embed32_lr00003_batch64.csv",
    #     "Two-32-00003-b64"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_TwoLayers_embed32_lr00003_batch128.csv",
    #     "Two-32-00003-b128"),
    #
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_TwoLayers_embed64_lr0001_batch64.csv",
    #     "Two-64-0001-b64"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_TwoLayers_embed64_lr0001_batch128.csv",
    #     "Two-64-0001-b128"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_TwoLayers_embed64_lr00003_batch64.csv",
    #     "Two-64-00003-b64"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_TwoLayers_embed64_lr00003_batch128.csv",
    #     "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_ThreeLayers_embed32_lr0001_batch64.csv",
    #     "Three-32-0001-b64"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_ThreeLayers_embed32_lr0001_batch128.csv",
    #     "Three-32-0001-b128"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_ThreeLayers_embed32_lr00003_batch64.csv",
    #     "Three-32-00003-b64"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_ThreeLayers_embed32_lr00003_batch128.csv",
    #     "Three-32-00003-b128"),
    #
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_ThreeLayers_embed64_lr0001_batch64.csv",
    #     "Three-64-0001-b64"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_ThreeLayers_embed64_lr0001_batch128.csv",
    #     "Three-64-0001-b128"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_ThreeLayers_embed64_lr00003_batch64.csv",
    #     "Three-64-00003-b64"),
    # (
    #     r"E:\Data\Recommend_test_100K_goodbooks(MLPwithGenres)\nngenres_ThreeLayers_embed64_lr00003_batch128.csv",
    #     "Three-64-00003-b128"),

    # # #diana genres movies 100k 18/12
    # # # #1 LAYER
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_movies(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),



    # # #diana pbr books 100k 18/12
    # # # #1 LAYER
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithGenres)\cleaned_CSVs\nnattr_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),

    # # #diana pbr books 100k 18/12
    # # # #1 LAYER
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\Bprnn_OneLayer_embed32_lr0001_batch64.csv" ,"One-32-0001-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_OneLayer_embed32_lr0001_batch128.csv" ,"One-32-0001-b128"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_OneLayer_embed32_lr00003_batch64.csv" ,"One-32-00003-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_OneLayer_embed32_lr00003_batch128.csv" ,"One-32-00003-b128"),
    #
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_OneLayer_embed64_lr0001_batch64.csv","One-64-0001-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_OneLayer_embed64_lr0001_batch128.csv","One-64-0001-128"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_OneLayer_embed64_lr00003_batch64.csv","One-64-00003-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_OneLayer_embed64_lr00003_batch128.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_TwoLayers_embed32_lr0001_batch64.csv","Two-32-0001-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_TwoLayers_embed32_lr0001_batch128.csv","Two-32-0001-b128"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_TwoLayers_embed32_lr00003_batch64.csv","Two-32-00003-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_TwoLayers_embed32_lr00003_batch128.csv","Two-32-00003-b128"),
    #
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_TwoLayers_embed64_lr0001_batch64.csv","Two-64-0001-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_TwoLayers_embed64_lr0001_batch128.csv","Two-64-0001-b128"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_TwoLayers_embed64_lr00003_batch64.csv","Two-64-00003-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_TwoLayers_embed64_lr00003_batch128.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_ThreeLayers_embed32_lr0001_batch64.csv","Three-32-0001-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_ThreeLayers_embed32_lr0001_batch128.csv","Three-32-0001-b128"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_ThreeLayers_embed32_lr00003_batch64.csv","Three-32-00003-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_ThreeLayers_embed32_lr00003_batch128.csv","Three-32-00003-b128"),
    #
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_ThreeLayers_embed64_lr0001_batch64.csv","Three-64-0001-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_ThreeLayers_embed64_lr0001_batch128.csv","Three-64-0001-b128"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_ThreeLayers_embed64_lr00003_batch64.csv","Three-64-00003-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_test_100K_goodbooks(MLPwithBPR)\cleaned_CSVs\BPRnn_ThreeLayers_embed64_lr00003_batch128.csv","Three-64-00003-b128"),

    # # #diana val  100k movies with pbr 18/12
    # # # #1 LAYER
    # (r"E:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr0001_batch128.csv" ,"One-32-0001-b128"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr00003_batch64.csv" ,"One-32-00003-b64"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr00003_batch128.csv" ,"One-32-00003-b128"),
    #
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr0001_batch64.csv","One-64-0001-b64"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr0001_batch128.csv","One-64-0001-128"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr00003_batch64.csv","One-64-00003-b64"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr00003_batch128.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr0001_batch64.csv","Two-32-0001-b64"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr0001_batch128.csv","Two-32-0001-b128"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr00003_batch64.csv","Two-32-00003-b64"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr00003_batch128.csv","Two-32-00003-b128"),
    #
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr0001_batch64.csv","Two-64-0001-b64"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr0001_batch128.csv","Two-64-0001-b128"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr00003_batch64.csv","Two-64-00003-b64"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr00003_batch128.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr0001_batch64.csv","Three-32-0001-b64"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr0001_batch128.csv","Three-32-0001-b128"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr00003_batch64.csv","Three-32-00003-b64"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr00003_batch128.csv","Three-32-00003-b128"),
    #
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr0001_batch64.csv","Three-64-0001-b64"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr0001_batch128.csv","Three-64-0001-b128"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr00003_batch64.csv","Three-64-00003-b64"),
    # (r"e:\Data\Output_Predictions_val_100k_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr00003_batch128.csv","Three-64-00003-b128"),

    # # #diana 1M movies with PBR Val 18/12
    # # # #1 LAYER
    #(
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_1M_movies(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),

    # # #diana 100K BOOKS with BPR 18/12
    # # # # # #1 LAYER
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"E:\Data\Output_Predictions_val_100K_goodbooks(MLPwithBPR)\BPRnn_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),



    # # #diana test for best hyper data 1m movies with Genres 18/12
    # # # #1 LAYER
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr0001_batch64.csv" ,"One-32-0001-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr0001_batch128.csv" ,"One-32-0001-b128"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr00003_batch64.csv" ,"One-32-00003-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr00003_batch128.csv" ,"One-32-00003-b128"),
    #
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr0001_batch64.csv","One-64-0001-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr0001_batch128.csv","One-64-0001-128"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr00003_batch64.csv","One-64-00003-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr00003_batch128.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr0001_batch64.csv","Two-32-0001-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr0001_batch128.csv","Two-32-0001-b128"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr00003_batch64.csv","Two-32-00003-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr00003_batch128.csv","Two-32-00003-b128"),
    #
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr0001_batch64.csv","Two-64-0001-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr0001_batch128.csv","Two-64-0001-b128"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr00003_batch64.csv","Two-64-00003-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr00003_batch128.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr0001_batch64.csv","Three-32-0001-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr0001_batch128.csv","Three-32-0001-b128"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr00003_batch64.csv","Three-32-00003-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr00003_batch128.csv","Three-32-00003-b128"),
    #
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr0001_batch64.csv","Three-64-0001-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr0001_batch128.csv","Three-64-0001-b128"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr00003_batch64.csv","Three-64-00003-b64"),
    # (r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_1M_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr00003_batch128.csv","Three-64-00003-b128"),

    # # #diana test for best hyper data 100K movies with Genres 18/12
    # # # #1 LAYER
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_movies(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),


    # #diana test best hyper params total data 100K BOOKS with BPR 18/12
    # # # # #1 LAYER
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"E:\Data\OUTPUT_datasets\NN\Output_Predictions_val_100k_goodbooks(MLPwithGenres)\NNattr_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),





##

    # # #diana total data 100K movies with BPR 18/12
    # # # #1 LAYER
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64_filtered.csv" ,"One-32-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128_filtered.csv" ,"One-32-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64_filtered.csv" ,"One-32-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128_filtered.csv" ,"One-32-00003-b128"),
    #
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64_filtered.csv","One-64-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128_filtered.csv","One-64-0001-128"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64_filtered.csv","One-64-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128_filtered.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64_filtered.csv","Two-32-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128_filtered.csv","Two-32-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64_filtered.csv","Two-32-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128_filtered.csv","Two-32-00003-b128"),
    #
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64_filtered.csv","Two-64-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128_filtered.csv","Two-64-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64_filtered.csv","Two-64-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128_filtered.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64_filtered.csv","Three-32-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128_filtered.csv","Three-32-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64_filtered.csv","Three-32-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128_filtered.csv","Three-32-00003-b128"),
    #
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64_filtered.csv","Three-64-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128_filtered.csv","Three-64-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64_filtered.csv","Three-64-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128_filtered.csv","Three-64-00003-b128"),

    # # #diana total data 1M movies with BPR 18/12
    # # # #1 LAYER
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64_filtered.csv",
    # "One-32-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128_filtered.csv",
    # "One-32-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64_filtered.csv",
    # "One-32-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128_filtered.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"E:\Data\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128_filtered.csv",
    # "One-64-0001-128"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64_filtered.csv",
    # "One-64-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128_filtered.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64_filtered.csv",
    # "Two-32-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128_filtered.csv",
    # "Two-32-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64_filtered.csv",
    # "Two-32-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128_filtered.csv",
    # "Two-32-00003-b128"),
    #
    # #(
    # #r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64_filtered.csv",
    # #"Two-64-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128_filtered.csv",
    # "Two-64-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64_filtered.csv",
    # "Two-64-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128_filtered.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64_filtered.csv",
    # "Three-32-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128_filtered.csv",
    # "Three-32-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64_filtered.csv",
    # "Three-32-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128_filtered.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64_filtered.csv",
    # "Three-64-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128_filtered.csv",
    # "Three-64-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64_filtered.csv",
    # "Three-64-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\Recommend_test_1m_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128_filtered.csv",
    # "Three-64-00003-b128"),

    # # #diana total data 100K BOOKS with BPR 18/12
    # # # #1 LAYER
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch64_filtered.csv" ,"One-32-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr0001_batch128_filtered.csv" ,"One-32-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch64_filtered.csv" ,"One-32-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed32_lr00003_batch128_filtered.csv" ,"One-32-00003-b128"),
    #
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_BPRnn_OneLayer_embed64_lr0001_batch64_filtered.csv","One-64-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr0001_batch128_filtered.csv","One-64-0001-128"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch64_filtered.csv","One-64-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_OneLayer_embed64_lr00003_batch128_filtered.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch64_filtered.csv","Two-32-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr0001_batch128_filtered.csv","Two-32-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch64_filtered.csv","Two-32-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed32_lr00003_batch128_filtered.csv","Two-32-00003-b128"),
    #
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64_filtered.csv","Two-64-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch128_filtered.csv","Two-64-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch64_filtered.csv","Two-64-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr00003_batch128_filtered.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch64_filtered.csv","Three-32-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr0001_batch128_filtered.csv","Three-32-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch64_filtered.csv","Three-32-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed32_lr00003_batch128_filtered.csv","Three-32-00003-b128"),
    #
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch64_filtered.csv","Three-64-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr0001_batch128_filtered.csv","Three-64-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch64_filtered.csv","Three-64-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\RecommendBPRnn_ThreeLayers_embed64_lr00003_batch128_filtered.csv","Three-64-00003-b128"),

    # # #diana test for best hyper data 100K movies with BPR 18/12
    # # # #1 LAYER
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed32_lr0001_batch64.csv" ,"One-32-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed32_lr0001_batch128.csv" ,"One-32-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed32_lr00003_batch64.csv" ,"One-32-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed32_lr00003_batch128.csv" ,"One-32-00003-b128"),
    #
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv","One-64-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed64_lr0001_batch128.csv","One-64-0001-128"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed64_lr00003_batch64.csv","One-64-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed64_lr00003_batch128.csv","One-64-00003-b128"),
    #
    # #2 LAYER
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed32_lr0001_batch64.csv","Two-32-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed32_lr0001_batch128.csv","Two-32-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed32_lr00003_batch64.csv","Two-32-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed32_lr00003_batch128.csv","Two-32-00003-b128"),
    #
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed64_lr0001_batch64.csv","Two-64-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed64_lr0001_batch128.csv","Two-64-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed64_lr00003_batch64.csv","Two-64-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed64_lr00003_batch128.csv","Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed32_lr0001_batch64.csv","Three-32-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed32_lr0001_batch128.csv","Three-32-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed32_lr00003_batch64.csv","Three-32-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed32_lr00003_batch128.csv","Three-32-00003-b128"),
    #
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed64_lr0001_batch64.csv","Three-64-0001-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed64_lr0001_batch128.csv","Three-64-0001-b128"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed64_lr00003_batch64.csv","Three-64-00003-b64"),
    # (r"E:\Data\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\Recommend_test_100K_movies_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed64_lr00003_batch128.csv","Three-64-00003-b128"),

    # # #diana test best hyper params total data 100K BOOKS with BPR 18/12
    # # # # #1 LAYER
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed32_lr0001_batch64.csv",
    # "One-32-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed32_lr0001_batch128.csv",
    # "One-32-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed32_lr00003_batch64.csv",
    # "One-32-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed32_lr00003_batch128.csv",
    # "One-32-00003-b128"),
    #
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommend_BPRnn_OneLayer_embed64_lr0001_batch64.csv",
    # "One-64-0001-b64"),
    # # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed64_lr0001_batch128.csv",
    # "One-64-0001-128"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed64_lr00003_batch64.csv",
    # "One-64-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_OneLayer_embed64_lr00003_batch128.csv",
    # "One-64-00003-b128"),
    #
    # # 2 LAYER
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed32_lr0001_batch64.csv",
    # "Two-32-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed32_lr0001_batch128.csv",
    # "Two-32-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed32_lr00003_batch64.csv",
    # "Two-32-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed32_lr00003_batch128.csv",
    # "Two-32-00003-b128"),
    #
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed64_lr0001_batch64.csv",
    # "Two-64-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed64_lr0001_batch128.csv",
    # "Two-64-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed64_lr00003_batch64.csv",
    # "Two-64-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_TwoLayers_embed64_lr00003_batch128.csv",
    # "Two-64-00003-b128"),
    #
    # # 3 LAYER
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed32_lr0001_batch64.csv",
    # "Three-32-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed32_lr0001_batch128.csv",
    # "Three-32-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed32_lr00003_batch64.csv",
    # "Three-32-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed32_lr00003_batch128.csv",
    # "Three-32-00003-b128"),
    #
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed64_lr0001_batch64.csv",
    # "Three-64-0001-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed64_lr0001_batch128.csv",
    # "Three-64-0001-b128"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed64_lr00003_batch64.csv",
    # "Three-64-00003-b64"),
    # (
    # r"E:\Data\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\Recommend_test_100K_goodbooks_TOTAL(MLPwithBPR)\_val_evalRecommendBPRnn_ThreeLayers_embed64_lr00003_batch128.csv",
    # "Three-64-00003-b128"),

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
    # r"E:\Data\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\Recommend_test_1M_movies_TOTAL(MLPwithBPR)\RecommendBPRnn_TwoLayers_embed64_lr0001_batch64_filtered.csv",
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





    ###### Kasia #######

    # #DPP - movies
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\2025-12-17_22-02-06\dpp_test_100000_jaccard_top_10.csv", "02, dpp_Jaccard"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\2025-12-17_22-02-06\dpp_test_100000_cosine_top_10.csv", "02, dpp_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\movies\2025-12-17_22-02-06\mf_test_100000_top_10.csv", "02, MF"),




    # # #DPP - books
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\books\2025-12-17_21-43-05\dpp_test_100000_jaccard_top_10.csv", "43 dpp_Jaccard"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\books\2025-12-17_21-43-05\dpp_test_100000_cosine_top_10.csv", "43 dpp_cosine"),
    # (r"C:\Users\Jacob\Documents\GitHub\P5\src\datasets\datasets_to_analyse\kasia_resultater\books\2025-12-17_21-43-05\mf_test_100000_top_10.csv", "43MF"),




]

