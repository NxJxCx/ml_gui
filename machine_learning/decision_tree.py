from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

from .base import MLBase
from .util import get_image_data_from_plot, encode_base64, map_list_json_compatible


def get_plots_by_instance(ml):
    if isinstance(ml, DecisionTreeClassification):
        return {
            "learning_curve": encode_base64(ml.plot_learning_curve()),
            "class_distribution": encode_base64(ml.plot_class_distribution()),
            "confusion_matrix": encode_base64(ml.plot_confusion_matrix()),
            "decision_tree": encode_base64(ml.plot_decision_tree()),
            "feature_importance": encode_base64(ml.plot_feature_importance()),
        }
    elif isinstance(ml, DecisionTreeRegression):
        return {
            "learning_curve": encode_base64(ml.plot_learning_curve()),
            "regression_tree": encode_base64(ml.plot_regression_tree()),
            "feature_importance": encode_base64(ml.plot_feature_importance()),
        }


class DecisionTreeClassification(MLBase):
    """
    DecisionTreeClassification Class for Decision Tree Classifier

    This module provides a `Classification` class that encapsulates the functionality
    to train, evaluate, and visualize a Decision Tree Classifier using scikit-learn.
    It includes methods for configuring the model, training it, evaluating its performance,
    and plotting various visualizations related to the model and data.

    Usage:
    1. Create an instance of the `Classification` class with the dataset, feature columns, and target column.
    2. Configure the training parameters using `configure_training()`.
    3. Train the model using `train_model()`.
    4. Evaluate the model using `evaluate_trained_model()`.
    5. Generate plots using the provided plotting methods.

    Example:
    ```python
    from your_module import Classification

    # Sample dataset
    data = {
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0]
    }

    # Initialize the classifier
    classifier = Classification(data, column_features=['feature1', 'feature2'], column_target='target')

    # Configure training
    classifier.configure_training()

    # Train the model
    classifier.train_model()

    # Evaluate the model
    results = classifier.evaluate_trained_model()
    ```
    """

    _algo_name = "Decision Tree Classification"
    _algorithm = "dtclassifier"

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
        self._algo: Optional[DecisionTreeClassifier] = None
        super().__init__(dataset, column_features, column_target)

    def set_column_features(self, column_features: Union[List[str], Tuple[str]]):
        """
        Set the feature columns for the classifier.

        :param column_features: A list or tuple of feature column names.
        """
        self._features = list(column_features)
        self.features = self.data[self._features]

    def configure_training(
        self,
        criterion: Literal["entropy", "gini"] = "gini",
        random_state: Optional[int] = None,
        test_size: float = 0.25,
        train_size: Optional[float] = None,
        max_depth: Optional[int] = None,
        min_samples_split: Optional[int] = 2,
        min_samples_leaf: Optional[int] = 1,
        max_features: Optional[Literal["auto", "sqrt", "log2"]] = None,
        splitter: Literal["best", "random"] = "best",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        shuffle: bool = True,
        stratify: Optional[object] = None,
        **kwargs
    ):
        """
        Configure Decision Tree Classifier for training

        :param criterion: The function to measure the quality of a split. Supported: 'gini' or 'entropy'.
        :param random_state: Controls the randomness of the estimator.
        :param test_size: The proportion of the data to be used for testing.
        :param train_size: The proportion of the data to be used for training.
        :param max_depth: The maximum depth of the tree.
        :param min_samples_split: The minimum number of samples required to split an internal node.
        :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
        :param max_features: The number of features to consider when looking for the best split.
        :param splitter: The strategy used to split at each node. Supported: 'best' or 'random'.
        :param max_leaf_nodes: The maximum number of leaf nodes in the tree.
        :param min_impurity_decrease: A node will be split if this split induces a decrease of impurity greater than or equal to this value.
        :param shuffle: Whether to shuffle the data before splitting. Default is True.
        :param stratify: Ensures that the split maintains the same distribution of classes as in the original data. Default is None.
        """
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            self.target,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
        # Create the DecisionTreeClassifier model with the specified hyperparameters
        self._algo = DecisionTreeClassifier(
            criterion=criterion,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            splitter=splitter,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
        )
        return super().configure_training(
            criterion=criterion,
            random_state=random_state,
            test_size=test_size,
            train_size=train_size,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            splitter=splitter,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            shuffle=shuffle,
            stratify=stratify,
            **kwargs
        )

    def evaluate_trained_model(self) -> Dict[str, Any]:
        """
        Evaluate the trained model on the test data.

        :return: A tuple containing test accuracy, training accuracy, cross-validation scores,
                classification reports for test and training data, confusion matrix,
                feature importance, and class distribution.
        :rtype: tuple
        """
        # Evaluate on test data
        self.y_pred = self._algo.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        # Training accuracy
        self.y_train_pred = self._algo.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, self.y_train_pred)
        cv_scores = cross_val_score(self._algo, self.features, self.target, cv=5)
        clf_report_test = classification_report(self.y_test, self.y_pred)
        clf_report_train = classification_report(self.y_train, self.y_train_pred)
        conf_matrix = confusion_matrix(self.y_test, self.y_pred, labels=self._algo.classes_)
        feature_importance = self._algo.feature_importances_
        lc = learning_curve(self._algo, self.features, self.target, cv=5, scoring="accuracy", n_jobs=-1)
        class_distribution = (self.data["Recommended_Strand"].value_counts(),)
        return {
            "test_accuracy": accuracy,  # test accuracy
            "training_accuracy": train_accuracy,  # training accuracy
            "cross_validation_scores": cv_scores,  # cross-validation scores
            "classification_report_test_data": clf_report_test,  # classification report on test data
            "classification_report_train_data": clf_report_train,  # classification report on training data
            "confusion_matrix": conf_matrix,  # confusion matrix
            "feature_importance": feature_importance,  # feature importance
            "learning_curve": lc,  # (train_sizes, train_scores, test_scores) [learning curve],
            "class_distribution": class_distribution,  # class distribution
        }

    def _plot_decision_tree(self):
        plt.figure(figsize=(50, 25))
        plot_tree(self._algo, filled=True, feature_names=self.features.columns, class_names=self._algo.classes_)

    def plot_decision_tree(self) -> bytes:
        """
        Plot the decision tree and capture the image data.

        :return: Image data of the decision tree plot.
        :rtype: bytes
        """
        data = get_image_data_from_plot(self._plot_decision_tree)
        plt.close()
        return data

    def _plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred, labels=self._algo.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self._algo.classes_)
        disp.plot(cmap="viridis")

    def plot_confusion_matrix(self) -> bytes:
        """
        Plot the confusion matrix and capture the image data.

        :return: Image data of the confusion matrix plot.
        :rtype: bytes
        """
        data = get_image_data_from_plot(self._plot_confusion_matrix)
        plt.close()
        return data

    def _plot_feature_importance(self):
        feature_importances = self._algo.feature_importances_
        plt.bar(self.features.columns, feature_importances)
        plt.xticks(rotation=45)
        plt.title("Feature Importance")

    def plot_feature_importance(self) -> bytes:
        """
        Plot the feature importance and capture the image data.

        :return: Image data of the feature importance plot.
        :rtype: bytes
        """
        data = get_image_data_from_plot(self._plot_feature_importance)
        plt.close()
        return data

    def _plot_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(
            self._algo, self.features, self.target, cv=5, scoring="accuracy", n_jobs=-1
        )
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Train Accuracy")
        plt.plot(train_sizes, test_scores.mean(axis=1), label="Test Accuracy")
        plt.legend()
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy")
        plt.title("Learning Curve")

    def plot_learning_curve(self) -> bytes:
        """
        Plot the learning curve and capture the image data.

        :return: Image data of the learning curve plot.
        :rtype: bytes
        """

        data = get_image_data_from_plot(self._plot_learning_curve)
        plt.close()
        return data

    def _plot_class_distribution(self):
        self.data[self._target].value_counts().plot(kind="bar", title="Class Distribution")

    def plot_class_distribution(self) -> bytes:
        """
        Plot the class distribution and capture the image data.

        :return: Image data of the class distribution plot.
        :rtype: bytes
        """
        data = get_image_data_from_plot(self._plot_class_distribution)
        plt.close()
        return data


class DecisionTreeRegression(MLBase):
    """
    DecisionTreeRegression Class for Decision Tree Regressor

    This module provides a `Regression` class that encapsulates the functionality
    to train, evaluate, and visualize a Decision Tree Regressor using scikit-learn.
    It includes methods for configuring the model, training it, evaluating its performance,
    and plotting various visualizations related to the model and data.

    Usage:
    1. Create an instance of the `Regression` class with the dataset, feature columns, and target column.
    2. Configure the training parameters using `configure_training()`.
    3. Train the model using `train_model()`.
    4. Evaluate the model using `evaluate_trained_model()`.
    5. Generate plots using the provided plotting methods.

    Example:
    ```python
    from your_module import Regression

    # Sample dataset
    data = {
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [10, 15, 20]
    }

    # Initialize the regressor
    regressor = Regression(data, column_features=['feature1', 'feature2'], column_target='target')

    # Configure training
    regressor.configure_training()

    # Train the model
    regressor.train_model()

    # Evaluate the model
    results = regressor.evaluate_trained_model()

    """

    _algo_name = "Decision Tree Regression"
    _algorithm = "dtregressor"

    def __init__(
        self,
        dataset: Union[Dict[str, Any], pd.DataFrame],
        column_features: Union[List[str], Tuple[str]],
        column_target: Union[str, Union[List[str], Tuple[str]]],
    ):
        """
        Initialize the Regression class.

        :param dataset: A dictionary representing the dataset, where keys are column names and values are lists of column data.
        :param column_features: A list or tuple of feature column names to be used for training.
        :param column_target: The name or list of names of the target column/s for regression.
        """
        self._algo: Optional[DecisionTreeRegressor] = None
        super().__init__(dataset, column_features, column_target)

    def configure_training(
        self,
        random_state: Optional[int] = None,
        test_size: float = 0.25,
        train_size: Optional[float] = None,
        max_depth: Optional[int] = None,
        min_samples_split: Optional[int] = 2,
        min_samples_leaf: Optional[int] = 1,
        max_features: Optional[Literal["auto", "sqrt", "log2"]] = None,
        splitter: Literal["best", "random"] = "best",
        **kwargs
    ):
        """
        Configure Decision Tree Regressor for training.

        :param random_state: Controls the randomness of the estimator.
        :param test_size: The proportion of the data to be used for testing.
        :param train_size: The proportion of the data to be used for training.
        :param max_depth: The maximum depth of the tree.
        :param min_samples_split: The minimum number of samples required to split an internal node.
        :param min_samples_leaf: The minimum number of samples required to be at a leaf node.
        :param max_features: The number of features to consider when looking for the best split.
        :param splitter: The strategy used to split at each node. Supported: 'best' or 'random'.
        """
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            self.target,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )

        # Create the DecisionTreeRegressor model with the specified hyperparameters
        self._algo = DecisionTreeRegressor(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            splitter=splitter,
        )
        return super().configure_training(
            random_state=random_state,
            test_size=test_size,
            train_size=train_size,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            splitter=splitter,
            **kwargs
        )

    def evaluate_trained_model(self) -> Dict[str, Any]:
        """
        Evaluate the trained model on the test data.

        :return: A tuple containing mean squared error, R² score, and cross-validation scores.
        :rtype: tuple
        """
        # Evaluate on test data
        y_pred = self._algo.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        cv_scores = cross_val_score(self._algo, self.features, self.target, cv=5)
        return {
            "mean_squared_error": mse,  # Mean Squared Error
            "r2_score": r2,  # R² Score
            "cross_validation_scores": cv_scores,  # Cross-validation scores
        }

    def plot_regression_tree(self) -> bytes:
        """
        Plot the regression tree and capture the image data.

        :return: Image data of the regression tree plot.
        :rtype: bytes
        """

        def func():
            plt.figure(figsize=(50, 25))
            plot_tree(self._algo, filled=True, feature_names=self.features.columns)
            plt.close()

        return get_image_data_from_plot(func)

    def plot_learning_curve(self) -> bytes:
        """
        Plot the learning curve and capture the image data.

        :return: Image data of the learning curve plot.
        :rtype: bytes
        """

        def func():
            train_sizes, train_scores, test_scores = learning_curve(
                self._algo, self.features, self.target, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
            )
            plt.plot(train_sizes, train_scores.mean(axis=1), label="Train MSE")
            plt.plot(train_sizes, test_scores.mean(axis=1), label="Test MSE")
            plt.legend()
            plt.xlabel("Training Set Size")
            plt.ylabel("Mean Squared Error")
            plt.title("Learning Curve")
            plt.close()

        return get_image_data_from_plot(func)

    def plot_feature_importance(self) -> bytes:
        """
        Plot the feature importance and capture the image data.

        :return: Image data of the feature importance plot.
        :rtype: bytes
        """

        def func():
            feature_importances = self._algo.feature_importances_
            plt.bar(self.features.columns, feature_importances)
            plt.xticks(rotation=45)
            plt.title("Feature Importance")
            plt.close()

        return get_image_data_from_plot(func)
