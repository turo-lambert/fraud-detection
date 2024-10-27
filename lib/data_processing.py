"""Data processing functions for the project."""
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """A class to handle feature engineering for the Fraud Detection Model."""

    def __init__(
        self,
        cols_to_drop: list[str],
        categorical_cols: list[str],
        claim_occured_col: str,
        claim_reported_col: str,
        types_mapping: Optional[dict[str, str]] = None,
    ) -> None:
        """Initializes the FeatureEngineer class.

        Args:
            cols_to_drop (list[str]): List of columns to drop if they are not useful for modeling.
            categorical_cols (list[str]): List of categorical columns to be encoded.
            claim_occured_col (str): Column name for the claim occurrence date.
            claim_reported_col (str): Column name for the claim reported date
            types_mapping (dict[str, str]): Mapping of column names to their data types.
        """
        self.cols_to_drop = cols_to_drop
        self.categorical_cols = categorical_cols
        self.claim_occured_col = claim_occured_col
        self.claim_reported_col = claim_reported_col
        self.types_mapping = types_mapping
        self.label_encoders = {}  # To store label encoders for each categorical column
        self.is_fitted = False  # Indicates if the instance has been fitted

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """Fits LabelEncoders on categorical columns.

        Args:
            X (pd.DataFrame): The input DataFrame.
            y (Optional[pd.Series]): Target variable, not used here.

        Returns:
            self
        """
        # Fit LabelEncoders for each categorical column
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))  # Convert to string to handle any unexpected NaNs or non-string categories
            self.label_encoders[col] = le

        self.is_fitted = True  # Set to True after fitting is complete
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies feature transformations to the input DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        if not self.is_fitted:
            raise ValueError(
                "This FeatureEngineer instance is not fitted yet. Please call 'fit' with appropriate data."
            )

        X = X.copy()

        # Step 1: Drop irrelevant columns
        X = self._cols_to_drop(X)

        # Step 2: Process dates and add derived features
        X = self._process_dates(X)

        # Step 3: Encode categorical variables
        X = self._encode_categorical(X)

        # Step 4: Convert data types
        X = self._convert_types(X)

        return X

    def _cols_to_drop(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drops unnecessary columns from the DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame without specified columns.
        """
        return X.drop(columns=self.cols_to_drop, errors="ignore")

    def _process_dates(self, X: pd.DataFrame) -> pd.DataFrame:
        """Processes date columns and adds new time-related features.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with date-related features.
        """

        def convert_to_date(date_int: int) -> datetime:
            """Convert integer format YYYYMMDD to a datetime object."""
            return datetime.strptime(str(date_int), "%Y%m%d") if pd.notna(date_int) else np.nan

        # Convert dates to datetime
        X[self.claim_occured_col] = X[self.claim_occured_col].apply(convert_to_date)
        X[self.claim_reported_col] = X[self.claim_reported_col].apply(convert_to_date)

        # Calculate days to report and add month-based features
        X["days_to_report"] = (X[self.claim_reported_col] - X[self.claim_occured_col]).dt.days
        X["claim_occurred_day"] = X[self.claim_occured_col].dt.day
        X["claim_occurred_month"] = X[self.claim_occured_col].dt.month
        X["claim_occurred_year"] = X[self.claim_occured_col].dt.year
        X["claim_reported_day"] = X[self.claim_reported_col].dt.day
        X["claim_reported_month"] = X[self.claim_reported_col].dt.month
        X["claim_reported_year"] = X[self.claim_reported_col].dt.year

        # Drop the original date columns
        X = X.drop(columns=[self.claim_occured_col, self.claim_reported_col], errors="ignore")
        return X

    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encodes categorical variables using LabelEncoder.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with label-encoded categorical variables.
        """
        for col in self.categorical_cols:
            if col in X:
                # Apply label encoding
                X[col] = self.label_encoders[col].transform(
                    X[col].astype(str)
                )  # Convert to string to handle NaNs or unexpected types
        return X

    def _convert_types(self, X: pd.DataFrame) -> pd.DataFrame:
        """Converts data types of columns to appropriate types.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with converted data types.
        """
        # Convert all columns according to the expected types
        if self.types_mapping is not None:
            for col, dtype in self.types_mapping.items():
                if col in X:
                    X[col] = X[col].astype(dtype)

        return X
