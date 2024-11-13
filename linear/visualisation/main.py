from linear.dataset.data import get_data
from linear.dataset.select_data import DataProcessor
from linear.hyperparameter_selection.hyperparameter import HyperparameterSelection
from linear.utils.training_utils import tmp

data = get_data()
repo = DataProcessor(data, "Class")
x_train = repo.get("x", "train")
x_test = repo.get("x", "test")
y_train = repo.get("y", "train")
y_test = repo.get("y", "test")

best_alpha_logistic = HyperparameterSelection.get_best_alpha_logistic(x_train, y_train, x_test, y_test)
best_c_svm = HyperparameterSelection.get_best_c_svm(x_train, y_train, x_test, y_test)
best_kernel_svm = HyperparameterSelection.get_best_kernel_svm(x_train, y_train, x_test, y_test)
best_alpha_ridge = HyperparameterSelection.get_best_alpha_ridge(x_train, y_train, x_test, y_test)

tmp(x_train, y_train, x_test, y_test, best_alpha_logistic, best_c_svm, best_kernel_svm, best_alpha_ridge)
