import pandas as pd

class DataHandler:
    def __init__(self, ground_truth_path, predictions_path):
        # Load data
        gt = pd.read_csv(ground_truth_path)
        pred = pd.read_csv(predictions_path)

        # FIX: Convert IDs to strings immediately
        id_columns = ['userId', 'title']
        for col in id_columns:
            if col in gt.columns:
                gt[col] = gt[col].astype(str)
            if col in pred.columns:
                pred[col] = pred[col].astype(str)

        # Convert to RecTools format
        self.ground_truth = self._to_rectools_format(gt, is_ground_truth=True)
        self.predictions = self._to_rectools_format(pred, is_ground_truth=False)

        # Validate columns
        self._validate_columns()

        # Prepare RecTools data structures
        self.interactions = self._prepare_interactions()
        self.recommendations = self._prepare_recommendations()

        # Store full dataset for RMSE/MAE
        self.full_interactions = self.interactions.copy()

    def _to_rectools_format(self, df, is_ground_truth):
        """Convert any DataFrame to RecTools standard format"""
        df = df.copy()
        column_map = {}
        if 'userId' in df.columns:
            column_map['userId'] = 'user_id'
        if 'title' in df.columns:
            column_map['title'] = 'item_id'

        if is_ground_truth and 'rating' in df.columns:
            column_map['rating'] = 'weight'
        elif not is_ground_truth:
            if 'rating_pred' in df.columns:
                column_map['rating_pred'] = 'weight'
            elif 'rating' in df.columns:
                column_map['rating'] = 'weight'

        return df.rename(columns=column_map)

    def _validate_columns(self):
        """Ensure required columns exist for RecTools"""
        required = ['user_id', 'item_id', 'weight']
        for df, name in [(self.ground_truth, "Ground Truth"),
                         (self.predictions, "Predictions")]:
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"{name} missing columns: {missing}")

    def _prepare_interactions(self):
        """Create interactions DataFrame (ground truth)"""
        interactions = self.ground_truth[['user_id', 'item_id', 'weight']].copy()
        return interactions

    def _prepare_recommendations(self):
        """Create recommendations DataFrame with ranks"""
        recos = self.predictions[['user_id', 'item_id', 'weight']].copy()
        recos = recos.sort_values(['user_id', 'weight'], ascending=[True, False])
        recos['rank'] = recos.groupby('user_id').cumcount() + 1
        return recos