import numpy as np
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from linear.classifier.svm_classifier import SVM
from linear.dataset.data import accuracy_metric
from linear.regression.logistic_regression_gd import LogisticRegressionGD
from linear.regression.ridge_regression import RidgeRegression


def train_and_evaluate_iteration(train_size, model_type, x_train, y_train, x_test, y_test, best_alpha_logistic,
                                 best_c_svm, best_kernel_svm):
    x_train_subset, _, y_train_subset, _ = train_test_split(x_train, y_train, train_size=train_size, random_state=None)

    if model_type == 'log_reg':
        log_reg = LogisticRegressionGD(alpha=best_alpha_logistic, iterations=1000, penalty='l2')
        log_reg.fit(x_train_subset, y_train_subset)
        train_score = accuracy_metric(y_train_subset, log_reg.predict(x_train_subset))
        test_score = accuracy_metric(y_test, log_reg.predict(x_test))

    elif model_type == 'svm':
        svm = SVM(C=best_c_svm, alpha=0.01, iterations=1000, kernel=best_kernel_svm)
        svm.fit(x_train_subset, 2 * y_train_subset - 1)
        train_score = accuracy_metric(2 * y_train_subset - 1, svm.predict(x_train_subset))
        test_score = accuracy_metric(2 * y_test - 1, svm.predict(x_test))

    return train_score, test_score


def plot_learning_curve_with_intervals(train_sizes, train_scores, test_scores, title, ridge_model, X_test, y_test,
                                       add_ridge_predictions=False):
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Train (Accuracy)')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')

    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Test (Accuracy)')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

    if add_ridge_predictions:
        ridge_predictions = ridge_model.predict(X_test)
        ridge_test_accuracy = accuracy_metric(y_test, ridge_predictions)
        plt.axhline(y=ridge_test_accuracy, color='b', linestyle='--', label='Ridge Test Accuracy')

    plt.title(title)
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def sequential_handling(
        train_sizes,
        n_iterations,
        x_train,
        y_train,
        x_test,
        y_test,
        best_alpha_logistic,
        best_c_svm,
        best_kernel_svm
):
    train_scores_log_reg = []
    test_scores_log_reg = []
    train_scores_svm = []
    test_scores_svm = []

    for train_size in train_sizes:
        log_reg_train_scores = []
        log_reg_test_scores = []
        svm_train_scores = []
        svm_test_scores = []

        for _ in range(n_iterations):
            train_score, test_score = train_and_evaluate_iteration(train_size, 'log_reg', x_train, y_train, x_test,
                                                                   y_test,
                                                                   best_alpha_logistic, best_c_svm, best_kernel_svm)
            log_reg_train_scores.append(train_score)
            log_reg_test_scores.append(test_score)

            train_score, test_score = train_and_evaluate_iteration(train_size, 'svm', x_train, y_train, x_test, y_test,
                                                                   best_alpha_logistic, best_c_svm, best_kernel_svm)
            svm_train_scores.append(train_score)
            svm_test_scores.append(test_score)

        train_scores_log_reg.append(log_reg_train_scores)
        test_scores_log_reg.append(log_reg_test_scores)
        train_scores_svm.append(svm_train_scores)
        test_scores_svm.append(svm_test_scores)

    return train_scores_log_reg, test_scores_log_reg, train_scores_svm, test_scores_svm


def parallel_handling(
        train_sizes,
        n_iterations,
        x_train,
        y_train,
        x_test,
        y_test,
        best_alpha_logistic,
        best_c_svm,
        best_kernel_svm
):
    train_scores_log_reg = []
    test_scores_log_reg = []
    train_scores_svm = []
    test_scores_svm = []

    for train_size in train_sizes:
        results_log_reg = Parallel(n_jobs=-1)(
            delayed(train_and_evaluate_iteration)(train_size, 'log_reg', x_train, y_train, x_test, y_test,
                                                  best_alpha_logistic, best_c_svm, best_kernel_svm) for _ in
            range(n_iterations))
        log_reg_train_scores, log_reg_test_scores = zip(*results_log_reg)

        results_svm = Parallel(n_jobs=-1)(
            delayed(train_and_evaluate_iteration)(train_size, 'svm', x_train, y_train, x_test, y_test,
                                                  best_alpha_logistic,
                                                  best_c_svm, best_kernel_svm) for _ in range(n_iterations))
        svm_train_scores, svm_test_scores = zip(*results_svm)

        train_scores_log_reg.append(log_reg_train_scores)
        test_scores_log_reg.append(log_reg_test_scores)
        train_scores_svm.append(svm_train_scores)
        test_scores_svm.append(svm_test_scores)

    return train_scores_log_reg, test_scores_log_reg, train_scores_svm, test_scores_svm


def tmp(x_train, y_train, x_test, y_test, best_alpha_logistic, best_c_svm, best_kernel_svm, best_alpha_ridge,
        type_of_handling):
    train_sizes = np.linspace(0.1, 0.9, num=5)
    n_iterations = 10

    ridge_model = RidgeRegression(alpha=best_alpha_ridge)
    ridge_model.fit(x_train, y_train)

    if type_of_handling == 'parallel':
        train_scores_log_reg, test_scores_log_reg, train_scores_svm, test_scores_svm = parallel_handling(
            train_sizes,
            n_iterations,
            x_train,
            y_train,
            x_test,
            y_test,
            best_alpha_logistic,
            best_c_svm,
            best_kernel_svm
        )
    else:
        train_scores_log_reg, test_scores_log_reg, train_scores_svm, test_scores_svm = sequential_handling(
            train_sizes,
            n_iterations,
            x_train,
            y_train,
            x_test,
            y_test,
            best_alpha_logistic,
            best_c_svm,
            best_kernel_svm
        )

    plot_learning_curve_with_intervals(train_sizes, train_scores_log_reg, test_scores_log_reg,
                                       'Learning Curve for Logistic Regression with Intervals', ridge_model, x_test,
                                       y_test)
    plot_learning_curve_with_intervals(train_sizes, train_scores_svm, test_scores_svm,
                                       'Learning Curve for SVM with Intervals', ridge_model, x_test, y_test, True)
