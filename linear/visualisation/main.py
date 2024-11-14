from linear.dataset.axis_repo import DataRepoImpl
from linear.hyperparameter_selection.cli_render import show_best_params
from linear.hyperparameter_selection.hyperparameter import HyperparameterSelection
from linear.utils.training_utils import tmp

repo = DataRepoImpl
x_train = repo.get_axis("x", "train")
x_test = repo.get_axis("x", "test")
y_train = repo.get_axis("y", "train")
y_test = repo.get_axis("y", "test")

best_alpha_logistic = HyperparameterSelection.get_best_alpha_logistic(x_train, y_train, x_test, y_test)
best_c_svm = HyperparameterSelection.get_best_c_svm(x_train, y_train, x_test, y_test)
best_kernel_svm = HyperparameterSelection.get_best_kernel_svm(x_train, y_train, x_test, y_test)
best_alpha_ridge = HyperparameterSelection.get_best_alpha_ridge(x_train, y_train, x_test, y_test)

type_of_handling = "parallel"

show_best_params(x_train, y_train, x_test, y_test)
tmp(
    x_train,
    y_train,
    x_test,
    y_test,
    best_alpha_logistic,
    best_c_svm,
    best_kernel_svm,
    best_alpha_ridge,
    type_of_handling
)
