import pandas as pd
import os
import numpy as np

#File paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
test_path = os.path.join(base_path, "datasets", "ratings_test_titles.csv")
mf_path = os.path.join(base_path, "datasets", "mf_test_predictions.csv")

#load data
test = pd.read_csv(test_path)
mf = pd.read_csv(mf_path)

#ground truth dic
ground_truth = (
    test.groupby("userId")["movie"]
    .apply(list)
    .to_dict()
)

#get top 10 user predictions
N = 10

topn_mf = (
    mf.groupby("userId", group_keys=False)
      .apply(lambda x: x.sort_values("mf_score", ascending=False).head(N))
      .reset_index(drop=True)
)

pred_mf = (
    topn_mf.groupby("userId")["movie"]
    .apply(list)
    .to_dict()
)


#precision and recall function

def precision_recall_at_k(preds, ground_truth, k):
    precisions = []
    recalls = []

    for user, pred_items in preds.items():
        if user not in ground_truth:
            continue

        true_items = set(ground_truth[user])
        pred_items = set(pred_items[:k])

        tp = len(true_items & pred_items)
        precisions.append(tp / k if k > 0 else 0)
        recalls.append(tp / len(true_items) if len(true_items) > 0 else 0)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    return avg_precision, avg_recall

# Compute metrics
precision, recall = precision_recall_at_k(pred_mf, ground_truth, N)

#results
print("Matrix Factorization Evaluation (Top 10)")
print(f"Precision@{N}: {precision:.4f}")
print(f"Recall@{N}:    {recall:.4f}")
print(topn_mf.head(10))

user = 1
print("Ground truth titles:", ground_truth[user])

print("Predicted titles:", pred_mf[user])

true_set = set(ground_truth[user])
pred_set = set(pred_mf[user])

print("Intersection:", true_set & pred_set)

