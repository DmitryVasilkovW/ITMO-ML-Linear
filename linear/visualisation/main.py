from linear.dataset.data import get_data
from linear.dataset.select_data import DataProcessor
from linear.hyperparameter_selection.hyperparameter import params
from linear.utils.training_utils import tmp

data = get_data()
repo = DataProcessor(data, "Class")
X_train = repo.get("x", "train")
X_test = repo.get("x", "test")
y_train = repo.get("y", "train")
y_test = repo.get("y", "test")

best_accuracy_ridge, best_alpha_ridge, best_accuracy_logistic, best_alpha_logistic, best_C_svm, best_kernel_svm = params(
    X_train,
    y_train, X_test,
    y_test)

tmp(X_train, y_train, X_test, y_test, best_alpha_logistic, best_C_svm, best_kernel_svm, best_alpha_ridge)
