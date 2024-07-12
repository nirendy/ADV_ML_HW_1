from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class Dataset(ABC):
    def __init__(self, config=None):
        self.config = config

    @abstractmethod
    def load_data(self) -> None:
        """Load the dataset into memory or prepare it for use."""
        pass

    @abstractmethod
    def get_train_loader(self) -> DataLoader:
        """Return the DataLoader for the training dataset.

        Returns:
            DataLoader: The DataLoader instance for the training data.
        """
        pass

    @abstractmethod
    def get_test_loader(self) -> DataLoader:
        """Return the DataLoader for the test dataset.

        Returns:
            DataLoader: The DataLoader instance for the test data.
        """
        pass

    def get_val_loader(self) -> DataLoader:
        """Return the DataLoader for the validation dataset.

        This method is optional and can be overridden if a validation set is available.

        Returns:
            DataLoader: The DataLoader instance for the validation data.
        """
        raise NotImplementedError("Validation loader is not implemented for this dataset.")
