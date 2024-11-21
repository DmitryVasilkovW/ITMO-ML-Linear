import numpy as np
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from linear.classifier.svm_classifier import SupportVectorMachine
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
            best_alpha_ridge,
            best_alpha_svm
    ):
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._best_alpha_logistic = best_alpha_logistic
        self._best_c_svm = best_c_svm
        self._best_kernel_svm = best_kernel_svm
        self._best_alpha_ridge = best_alpha_ridge
        self._best_alpha_svm = best_alpha_svm
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

        self._get_plot(train_scores_log_reg, test_scores_log_reg,
                       self._label_for_regression_plot, ridge_model, False)
        self._get_plot(train_scores_svm, test_scores_svm,
                       self._label_for_svm_classification_plot, ridge_model, True)

    def _train(self, train_size, model_type):
        x_train_subset, _, y_train_subset, _ = train_test_split(
            self._x_train,
            self._y_train,
            train_size=train_size,
            random_state=42
        )
        if model_type == 'reg':
            train_score, test_score = self._regression_score_handler(x_train_subset, y_train_subset)
        elif model_type == 'stoch_reg':
            train_score, test_score = self._stochastic_regression_score_handler(x_train_subset, y_train_subset)
        elif model_type == 'svm':
            train_score, test_score = self._svm_score_handler(x_train_subset, y_train_subset)

        return train_score, test_score

    def _regression_score_handler(self, x_train_subset, y_train_subset):
        reg = LogisticRegressionGD(alpha=self._best_alpha_logistic, iterations=1000, penalty='l2')
        reg.fit(x_train_subset, y_train_subset)
        train_score = accuracy_metric(y_train_subset, reg.predict(x_train_subset))
        test_score = accuracy_metric(self._y_test, reg.predict(self._x_test))

        return train_score, test_score

    def _stochastic_regression_score_handler(self, x_train_subset, y_train_subset):
        stoch_reg = LogisticRegressionSGD(alpha=self._best_alpha_logistic, penalty='l2')
        stoch_reg.fit(x_train_subset, y_train_subset)
        train_score = accuracy_metric(y_train_subset, stoch_reg.predict(x_train_subset))
        test_score = accuracy_metric(self._y_test, stoch_reg.predict(self._x_test))

        return train_score, test_score

    def _svm_score_handler(self, x_train_subset, y_train_subset):
        svm = SupportVectorMachine(c=self._best_c_svm, alpha=self._best_alpha_svm, iterations=1000,
                                   kernel=self._best_kernel_svm)
        svm.fit(x_train_subset, 2 * y_train_subset - 1)
        train_score = accuracy_metric(2 * y_train_subset - 1, svm.predict(x_train_subset))
        test_score = accuracy_metric(2 * self._y_test - 1, svm.predict(self._x_test))

        return train_score, test_score

    def _sequential_handling(self, train_sizes, n_iterations, type_of_regression):
        train_scores_reg = []
        test_scores_reg = []
        train_scores_svm = []
        test_scores_svm = []

        for train_size in train_sizes:
            reg_train_scores = []
            reg_test_scores = []
            svm_train_scores = []
            svm_test_scores = []

            for _ in range(n_iterations):
                train_score, test_score = self._train(train_size, type_of_regression)
                reg_train_scores.append(train_score)
                reg_test_scores.append(test_score)

                train_score, test_score = self._train(train_size, 'svm')
                svm_train_scores.append(train_score)
                svm_test_scores.append(test_score)

            train_scores_reg.append(reg_train_scores)
            test_scores_reg.append(reg_test_scores)
            train_scores_svm.append(svm_train_scores)
            test_scores_svm.append(svm_test_scores)

        return train_scores_reg, test_scores_reg, train_scores_svm, test_scores_svm

    def _parallel_handling(self, train_sizes, n_iterations, type_of_regression):
        train_scores_reg = []
        test_scores_reg = []
        train_scores_svm = []
        test_scores_svm = []

        for train_size in train_sizes:
            results_for_reg = Parallel(n_jobs=-1)(
                delayed(self._train)(train_size, type_of_regression)
                for _ in range(n_iterations))
            reg_train_scores, reg_test_scores = zip(*results_for_reg)

            results_for_svm = Parallel(n_jobs=-1)(
                delayed(self._train)(train_size, 'svm')
                for _ in range(n_iterations))
            svm_train_scores, svm_test_scores = zip(*results_for_svm)

            train_scores_reg.append(reg_train_scores)
            test_scores_reg.append(reg_test_scores)
            train_scores_svm.append(svm_train_scores)
            test_scores_svm.append(svm_test_scores)

        return train_scores_reg, test_scores_reg, train_scores_svm, test_scores_svm

    def _get_plot(self, train_scores, test_scores, title, ridge_model, whether_to_add_prediction_bridge):
        train_scores = np.array(train_scores)
        test_scores = np.array(test_scores)

        n_iterations = train_scores.shape[1]
        confidence_coef = 1.96

        train_mean = np.mean(train_scores, axis=0)
        test_mean = np.mean(test_scores, axis=0)

        train_std = np.std(train_scores, axis=0)
        test_std = np.std(test_scores, axis=0)
        train_stderr = train_std / np.sqrt(n_iterations)
        test_stderr = test_std / np.sqrt(n_iterations)

        train_conf_interval = confidence_coef * train_stderr
        test_conf_interval = confidence_coef * test_stderr

        iterations = range(n_iterations)

        plt.plot(iterations, train_mean, 'o-', color='b', label='Train (Accuracy)')
        plt.fill_between(iterations, train_mean - train_conf_interval, train_mean + train_conf_interval,
                         alpha=0.2, color='b', label='Train 95% CI')

        plt.plot(iterations, test_mean, 'o-', color='g', label='Test (Accuracy)')
        plt.fill_between(iterations, test_mean - test_conf_interval, test_mean + test_conf_interval,
                         alpha=0.2, color='g', label='Test 95% CI')

        if whether_to_add_prediction_bridge:
            ridge_predictions = ridge_model.predict(self._x_test)
            ridge_test_accuracy = accuracy_metric(self._y_test, ridge_predictions)
            plt.axhline(y=ridge_test_accuracy, color='b', linestyle='--', label='Ridge Test Accuracy')

        plt.title(title)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
