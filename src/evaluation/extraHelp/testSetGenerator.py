import pandas as pd

# Create test predictions for the first 5 users
# User 1: Recommend 2 movies they rated highly (5.0) and 1 they didn't rate
test_predictions = [
    # User 1 - Should have 2 hits (Platoon, RoboCop already rated 5.0) and 1 miss (new movie)
    {"userId": 1, "title": "Platoon (1986)", "rating": 4.8},
    {"userId": 1, "title": "RoboCop (1987)", "rating": 4.7},
    {"userId": 1, "title": "Terminator 2: Judgment Day (1991)", "rating": 4.5},

    # User 2 - Should have 2 hits (Snow White rated 5.0, Pocahontas rated 5.0) and 1 miss
    {"userId": 2, "title": "Snow White and the Seven Dwarfs (1937)", "rating": 4.9},
    {"userId": 2, "title": "Pocahontas (1995)", "rating": 4.8},
    {"userId": 2, "title": "Beauty and the Beast (1991)", "rating": 4.6},

    # User 3 - Should have 2 hits (Apollo 13 rated 5.0, Gladiator rated 5.0) and 1 miss
    {"userId": 3, "title": "Apollo 13 (1995)", "rating": 4.9},
    {"userId": 3, "title": "Gladiator (2000)", "rating": 4.8},
    {"userId": 3, "title": "Braveheart (1995)", "rating": 4.7},

    # User 4 - Should have 1 hit (Clerks rated 4.0) and 2 misses
    {"userId": 4, "title": "Clerks (1994)", "rating": 4.5},
    {"userId": 4, "title": "Mallrats (1995)", "rating": 4.2},
    {"userId": 4, "title": "Chasing Amy (1997)", "rating": 4.0},

    # User 5 - Should have 2 hits (Fugitive rated 4.0, Clear and Present Danger rated 4.0) and 1 miss
    {"userId": 5, "title": "Fugitive, The (1993)", "rating": 4.6},
    {"userId": 5, "title": "Clear and Present Danger (1994)", "rating": 4.5},
    {"userId": 5, "title": "Air Force One (1997)", "rating": 4.3},
]

# Create DataFrame
df_predictions = pd.DataFrame(test_predictions)

# Save to CSV
df_predictions.to_csv("test_predictions.csv", index=False)

print("Test predictions created!")
print(f"\nDataFrame shape: {df_predictions.shape}")
print(f"\nFirst few rows:")
print(df_predictions.head(10))

# Expected results summary (assuming threshold >= 4.0 for relevant)
print("\n" + "=" * 60)
print("EXPECTED EVALUATION RESULTS (if threshold = 4.0):")
print("=" * 60)
print("User 1: 2 hits (Platoon, RoboCop), 1 miss (Terminator 2)")
print("User 2: 2 hits (Snow White, Pocahontas), 1 miss (Beauty and the Beast)")
print("User 3: 2 hits (Apollo 13, Gladiator), 1 miss (Braveheart)")
print("User 4: 1 hit (Clerks), 2 misses (Mallrats, Chasing Amy)")
print("User 5: 2 hits (Fugitive, Clear and Present Danger), 1 miss (Air Force One)")
print("\nTotal: 9 hits out of 15 recommendations")
print("Overall Precision@3: 9/15 = 0.60")
print("Overall Recall@3: Will depend on how many relevant items each user has in ground truth")