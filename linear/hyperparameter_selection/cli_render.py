from linear.hyperparameter_selection.hyperparameter import HyperparameterSelection


def show_best_params(x_train, y_train, x_test, y_test):
    params = HyperparameterSelection
    best_alpha_ridge = params.get_best_alpha_ridge(x_train, y_train, x_test, y_test)
    best_accuracy_ridge = params.get_best_accuracy_ridge(x_train, y_train, x_test, y_test)
    best_alpha_logistic = params.get_best_alpha_logistic(x_train, y_train, x_test, y_test)
    best_accuracy_logistic = params.get_best_accuracy_logistic(x_train, y_train, x_test, y_test)
    best_c_svm = params.get_best_c_svm(x_train, y_train, x_test, y_test)
    best_kernel_svm = params.get_best_kernel_svm(x_train, y_train, x_test, y_test)
    best_accuracy_svm = params.get_best_accuracy_svm(x_train, y_train, x_test, y_test)
    best_alpha_svm = params.get_best_alpha_svm(x_train, y_train, x_test, y_test)

    _print_params(
        best_alpha_ridge,
        best_accuracy_ridge,
        best_alpha_logistic,
        best_accuracy_logistic,
        best_c_svm,
        best_kernel_svm,
        best_accuracy_svm,
        best_alpha_svm
    )


def _print_params(
        best_alpha_ridge,
        best_accuracy_ridge,
        best_alpha_logistic,
        best_accuracy_logistic,
        best_c_svm,
        best_kernel_svm,
        best_accuracy_svm,
        best_alpha_svm
):
    print(f'Best alpha for Ridge Regression: {best_alpha_ridge}')
    print(f'Best alpha for Logistic Regression: {best_alpha_logistic}')
    print(f'Best alpha for SVM: {best_alpha_svm}')
    print(f'Best C for SVM: {best_c_svm}')
    print(f'Best kernel: {best_kernel_svm}')
    print(f'Best accuracy for Ridge Regression: {best_accuracy_ridge:.2f}')
    print(f'Best accuracy for Logistic Regression: {best_accuracy_logistic:.2f}')
    print(f'Best accuracy for SVM: {best_accuracy_svm:.2f}')
