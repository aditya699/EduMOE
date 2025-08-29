from typing import Optional
from pydantic import BaseModel, Field
from datasets import load_dataset, DatasetDict


class DatasetConfig(BaseModel):
    """Configuration model for dataset loading."""
    dataset_name: str = Field(..., description="Name of the dataset to load (e.g., 'wikitext').")
    subset_name: str = Field(..., description="Subset/config name (e.g., 'wikitext-2-raw-v1').")


class DatasetLoader:
    """Wrapper class for loading HuggingFace datasets."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def load_data(self) -> DatasetDict:
        """Load dataset using HuggingFace."""
        dataset = load_dataset(self.config.dataset_name, self.config.subset_name)
        return dataset


config = DatasetConfig(dataset_name="wikitext", subset_name="wikitext-2-raw-v1")
loader = DatasetLoader(config)
dataset = loader.load_data()

print(dataset)
for i in range(10):
    print(f"Row {i}: {dataset['train'][i]['text']!r}")

