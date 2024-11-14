import numpy as np
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from linear.classifier.svm_classifier import SVM
from linear.dataset.data import accuracy_metric
from linear.regression.logistic_regression_gd import LogisticRegressionGD
from linear.regression.logistic_regression_sgd import LogisticRegressionSGD
from linear.regression.ridge_regression import RidgeRegression


class TrainingResults:
    def __init__(
            self,
            x_train,
            y_train,
            x_test,
            y_test,
            best_alpha_logistic,
            best_c_svm,
            best_kernel_svm,
            best_alpha_ridge
    ):
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._best_alpha_logistic = best_alpha_logistic
        self._best_c_svm = best_c_svm
        self._best_kernel_svm = best_kernel_svm
        self._best_alpha_ridge = best_alpha_ridge
        self._label_for_regression_plot = 'Learning Curve for Logistic Regression with Intervals'
        self._label_for_svm_classification_plot = 'Learning Curve for SVM with Intervals'

    def show_results(self, type_of_handling, type_of_regression):
        train_sizes = np.linspace(0.1, 0.9, num=5)
        n_iterations = 10

        ridge_model = RidgeRegression(alpha=self._best_alpha_ridge)
        ridge_model.fit(self._x_train, self._y_train)

        if type_of_handling == 'parallel':
            (train_scores_log_reg,
             test_scores_log_reg,
             train_scores_svm,
             test_scores_svm) = self._parallel_handling(train_sizes, n_iterations, type_of_regression)
        else:
            (train_scores_log_reg,
             test_scores_log_reg,
             train_scores_svm,
             test_scores_svm) = self._sequential_handling(train_sizes, n_iterations, type_of_regression)

        self._get_plot(train_sizes, train_scores_log_reg, test_scores_log_reg,
                       self._label_for_regression_plot, ridge_model)
        self._get_plot(train_sizes, train_scores_svm, test_scores_svm,
                       self._label_for_svm_classification_plot, ridge_model, True)

    def _train(self, train_size, model_type):
        x_train_subset, _, y_train_subset, _ = train_test_split(
            self._x_train,
            self._y_train,
            train_size=train_size,
            random_state=None
        )
        if model_type == 'reg':
            train_score, test_score = self._regression_score_handler(x_train_subset, y_train_subset)
        elif model_type == 'stoch_reg':
            train_score, test_score = self._stochastic_regression_score_handler(x_train_subset, y_train_subset)
        elif model_type == 'svm':
            train_score, test_score = self._svm_score_handler(x_train_subset, y_train_subset)

        return train_score, test_score

    def _regression_score_handler(self, x_train_subset, y_train_subset):
        log_reg = LogisticRegressionGD(alpha=self._best_alpha_logistic, iterations=1000, penalty='l2')
        log_reg.fit(x_train_subset, y_train_subset)
        train_score = accuracy_metric(y_train_subset, log_reg.predict(x_train_subset))
        test_score = accuracy_metric(self._y_test, log_reg.predict(self._x_test))

        return train_score, test_score

    def _stochastic_regression_score_handler(self, x_train_subset, y_train_subset):
        log_reg = LogisticRegressionSGD(alpha=self._best_alpha_logistic, penalty='l2')
        log_reg.fit(x_train_subset, y_train_subset)
        train_score = accuracy_metric(y_train_subset, log_reg.predict(x_train_subset))
        test_score = accuracy_metric(self._y_test, log_reg.predict(self._x_test))

        return train_score, test_score

    def _svm_score_handler(self, x_train_subset, y_train_subset):
        svm = SVM(C=self._best_c_svm, alpha=0.01, iterations=1000, kernel=self._best_kernel_svm)
        svm.fit(x_train_subset, 2 * y_train_subset - 1)
        train_score = accuracy_metric(2 * y_train_subset - 1, svm.predict(x_train_subset))
        test_score = accuracy_metric(2 * self._y_test - 1, svm.predict(self._x_test))

        return train_score, test_score

    def _sequential_handling(self, train_sizes, n_iterations, type_of_regression):
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
                train_score, test_score = self._train(train_size, type_of_regression)
                log_reg_train_scores.append(train_score)
                log_reg_test_scores.append(test_score)

                train_score, test_score = self._train(train_size, 'svm')
                svm_train_scores.append(train_score)
                svm_test_scores.append(test_score)

            train_scores_log_reg.append(log_reg_train_scores)
            test_scores_log_reg.append(log_reg_test_scores)
            train_scores_svm.append(svm_train_scores)
            test_scores_svm.append(svm_test_scores)

        return train_scores_log_reg, test_scores_log_reg, train_scores_svm, test_scores_svm

    def _parallel_handling(self, train_sizes, n_iterations, type_of_regression):
        train_scores_log_reg = []
        test_scores_log_reg = []
        train_scores_svm = []
        test_scores_svm = []

        for train_size in train_sizes:
            results_log_reg = Parallel(n_jobs=-1)(
                delayed(self._train)(train_size, type_of_regression)
                for _ in range(n_iterations))
            log_reg_train_scores, log_reg_test_scores = zip(*results_log_reg)

            results_svm = Parallel(n_jobs=-1)(
                delayed(self._train)(train_size, 'svm')
                for _ in range(n_iterations))
            svm_train_scores, svm_test_scores = zip(*results_svm)

            train_scores_log_reg.append(log_reg_train_scores)
            test_scores_log_reg.append(log_reg_test_scores)
            train_scores_svm.append(svm_train_scores)
            test_scores_svm.append(svm_test_scores)

        return train_scores_log_reg, test_scores_log_reg, train_scores_svm, test_scores_svm

    def _get_plot(self, train_sizes, train_scores, test_scores, title, ridge_model,
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
            ridge_predictions = ridge_model.predict(self._x_test)
            ridge_test_accuracy = accuracy_metric(self._y_test, ridge_predictions)
            plt.axhline(y=ridge_test_accuracy, color='b', linestyle='--', label='Ridge Test Accuracy')

        plt.title(title)
        plt.xlabel('Training Size')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
