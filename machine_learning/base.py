from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .typing import MatrixLike


class MLBase:
    _algo_name = ""
    _algorithm = ""

    def __init__(
        self,
        dataset: Union[Dict[str, Any], pd.DataFrame],
        column_features: Union[List[str], Tuple[str]],
        column_target: Union[List[str], Tuple[str]],
    ):
        """
        Initialize the Classification class.
        :param dataset: A dictionary representing the dataset, where keys are column names and values are lists of column data.
        :param column_features: A list or tuple of feature column names to be used for training.
        :param column_target: The name of the target column for classification.
        """
        # Dataset Input for training
        self.data = pd.DataFrame(dataset) if type(dataset) is dict else dataset
        self._features = list(column_features)
        self._target = list(column_target)
        self.features = self.data[self._features]
        self.target = self.data[self._target]
        self._algo: Optional[Any] = None
        self.X_train: Optional[MatrixLike | ArrayLike] = None
        self.X_test: Optional[MatrixLike | ArrayLike] = None
        self.y_train: Optional[MatrixLike | ArrayLike] = None
        self.y_test: Optional[MatrixLike | ArrayLike] = None
        self.y_pred: Optional[MatrixLike | ArrayLike] = None
        self.y_train_pred: Optional[MatrixLike | ArrayLike] = None
        self.training_time = 0
        self.hyperparameters = {}

    def set_column_features(self, column_features: Union[List[str], Tuple[str]]):
        """
        Set the feature columns for the regressor.

        :param column_features: A list or tuple of feature column names.
        """
        self._features = list(column_features)
        self.features = self.data[self._features]

    def set_column_target(self, column_target: Union[List[str], Tuple[str]]):
        """
        Set the target column for the regressor.

        :param column_target: The name or the list of names of the target column/s.
        """
        self._target = list(column_target)
        self.target = self.data[self._target]

    def configure_training(self, **hyperparameters):
        """
        Configure Decision Tree Classifier for training

        :param **hyperparameters: Arbitrary keyword arguments representing hyperparameters
                                    for the Machine Learning Algorithm

        """
        self.hyperparameters = hyperparameters

    def train_model(self) -> bool:
        """
        Train a Decision Tree Classifier using the training data.

        :return: `True` if the model was successfully trained, `False` if training data is missing.
        :rtype: bool
        """
        if self._algo is None or self.X_train is None or self.y_train is None:
            return False
        # start training time
        start_time = perf_counter()
        print("started training:", start_time)
        print("X_train")
        print(self.X_train)
        print("y_train")
        print(self.y_train)
        print("clf")
        print(self._algo)
        # Train the model
        self._algo.fit(self.X_train, self.y_train)
        # end training time
        self.training_time = perf_counter() - start_time
        print("end time:", self.training_time)
        return True

    def evaluate_trained_model(self) -> Dict[str, Any]:
        """
        Evaluate the trained model on the test data.

        :return: A dictionary containing test and training results
        :rtype: dict
        """
        return {}

    def predict(self, input_data: Optional[Union[MatrixLike, ArrayLike, Any]]) -> Optional[NDArray]:
        if self._algo is None:
            return None
        if type(input_data) is dict:
            input_data = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.Series):
            input_data = input_data.values.reshape(-1, 1)
        elif not type(input_data) is list and not type(input_data) is tuple and not isinstance(input_data, NDArray):
            return None
        else:
            input_data = np.atleast_2d(input_data)
        y_pred = self._algo.predict(input_data)
        return y_pred

    @property
    def algorithm(self):
        return self._algo_name

    @property
    def algorithm_key(self):
        return self._algorithm