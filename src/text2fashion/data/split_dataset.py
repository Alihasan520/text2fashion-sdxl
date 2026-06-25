from __future__ import annotations

from datasets import Dataset, DatasetDict


def create_train_validation_test_split(
    dataset: Dataset,
    validation_size: float = 0.10,
    test_size: float = 0.10,
    seed: int = 42,
) -> DatasetDict:
    """Create reproducible train, validation, and test splits."""
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must be positive.")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1.")

    temp_size = validation_size + test_size
    train_temp = dataset.train_test_split(test_size=temp_size, seed=seed)

    relative_test_size = test_size / temp_size
    val_test = train_temp["test"].train_test_split(test_size=relative_test_size, seed=seed)

    return DatasetDict(
        {
            "train": train_temp["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )
