"""
Part of Phase 7: train/validation/test split
------------------------------------------------
WHY STRATIFIED RANDOM: this is a static snapshot with no genuine time
ordering (UDI is just a row index, not a timestamp) and no aggregation
step needed — the same reasoning as Day 1, 7, and 9. Stratification
matters even more here than in those cases given the severity of the
imbalance (3.39%), to guarantee every split has enough failure examples
to evaluate on at all.
"""

from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.15, val_size=0.15, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
