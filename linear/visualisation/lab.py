from linear.dataset.axis_repo import DataRepoImpl
from linear.hyperparameter_selection.cli_render import show_best_params
from linear.hyperparameter_selection.hyperparameter import HyperparameterSelection
from linear.training.training_results import TrainingResults


def show_best_params_for_lab():
    repo = DataRepoImpl
    x_train = repo.get_axis("x", "train")
    x_test = repo.get_axis("x", "test")
    y_train = repo.get_axis("y", "train")
    y_test = repo.get_axis("y", "test")

    show_best_params(x_train, y_train, x_test, y_test)


def show_plots_for_lab():
    repo = DataRepoImpl
    x_train = repo.get_axis("x", "train")
    x_test = repo.get_axis("x", "test")
    y_train = repo.get_axis("y", "train")
    y_test = repo.get_axis("y", "test")

    best_alpha_logistic = HyperparameterSelection.get_best_alpha_logistic(x_train, y_train, x_test, y_test)
    best_c_svm = HyperparameterSelection.get_best_c_svm(x_train, y_train, x_test, y_test)
    best_kernel_svm = HyperparameterSelection.get_best_kernel_svm(x_train, y_train, x_test, y_test)
    best_alpha_ridge = HyperparameterSelection.get_best_alpha_ridge(x_train, y_train, x_test, y_test)
    best_alpha_svm = HyperparameterSelection.get_best_alpha_svm(x_train, y_train, x_test, y_test)

    type_of_handling = "parallel"
    type_of_regression = "reg"

    results = TrainingResults(
        x_train,
        y_train,
        x_test,
        y_test,
        best_alpha_logistic,
        best_c_svm,
        best_kernel_svm,
        best_alpha_ridge,
        best_alpha_svm,
    )

    results.show_results(type_of_handling, type_of_regression)
    