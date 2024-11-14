from linear.classifier.regression import RidgeRegression, LogisticRegressionGD
from linear.classifier.svm_classifier import SVM
from linear.dataset.data import accuracy_metric


class HyperparameterSelection:
    _best_accuracy_ridge = 0
    _best_alpha_ridge = 0
    _best_accuracy_logistic = 0
    _best_alpha_logistic = 0
    _best_accuracy_svm = 0
    _best_c_svm = 0
    _best_kernel_svm = ''

    _alphas_ridge = [0.01, 0.3, 1.0]
    _alphas_logistic = [0.01, 1.0]
    _kernels = ['linear', 'polynomial', 'rbf']
    _cs = [0.1, 0.3]

    @classmethod
    def _set_hyperparameters(cls, x_train, y_train, x_test, y_test):
        cls._clean()
        for alpha in cls._alphas_ridge:
            ridge = RidgeRegression(alpha=alpha)
            ridge.fit(x_train, y_train)
            y_pred = ridge.predict_classes(x_test)
            accuracy = accuracy_metric(y_test, y_pred)
            if accuracy > cls._best_accuracy_ridge:
                cls._best_accuracy_ridge = accuracy
                cls._best_alpha_ridge = alpha

        for alpha in cls._alphas_logistic:
            log_reg = LogisticRegressionGD(alpha=alpha, iterations=1000, penalty='l2')
            log_reg.fit(x_train, y_train)
            y_pred = log_reg.predict_classes(x_test)
            accuracy = accuracy_metric(y_test, y_pred)
            if accuracy > cls._best_accuracy_logistic:
                cls._best_accuracy_logistic = accuracy
                cls._best_alpha_logistic = alpha

        for C in cls._cs:
            for kernel in cls._kernels:
                svm = SVM(C=C, alpha=0.01, kernel=kernel)
                svm.fit(x_train, 2 * y_train - 1)
                y_pred = svm.predict(x_test)
                accuracy = accuracy_metric(y_test, (y_pred >= 0).astype(int))
                if accuracy > cls._best_accuracy_svm:
                    cls._best_accuracy_svm = accuracy
                    cls._best_c_svm = C
                    cls._best_kernel_svm = kernel

    @classmethod
    def _clean(cls):
        cls._best_accuracy_ridge = 0
        cls._best_alpha_ridge = 0
        cls._best_accuracy_logistic = 0
        cls._best_alpha_logistic = 0
        cls._best_accuracy_svm = 0
        cls._best_c_svm = 0
        cls._best_kernel_svm = ''

    @classmethod
    def set_alphas_ridge(cls, alphas_ridge):
        cls._clean()
        cls._alphas_ridge = alphas_ridge

    @classmethod
    def set_alphas_logistic(cls, alphas_logistic):
        cls._clean()
        cls._alphas_logistic = alphas_logistic

    @classmethod
    def set_kernels(cls, kernels):
        cls._clean()
        cls._kernels = kernels

    @classmethod
    def set_cs(cls, cs):
        cls._clean()
        cls._cs = cs

    @classmethod
    def reset_greed_of_parameters(cls):
        cls._clean()
        cls._alphas_ridge = [0.01, 0.3, 1.0]
        cls._alphas_logistic = [0.01, 1.0]
        cls._kernels = ['linear', 'polynomial', 'rbf']
        cls._cs = [0.1, 0.3]

    @classmethod
    def get_best_accuracy_ridge(cls, x_train, y_train, x_test, y_test):
        if cls._best_accuracy_ridge == 0:
            cls._set_hyperparameters(x_train, y_train, x_test, y_test)
        return cls._best_accuracy_ridge

    @classmethod
    def get_best_alpha_ridge(cls, x_train, y_train, x_test, y_test):
        if cls._best_alpha_ridge == 0:
            cls._set_hyperparameters(x_train, y_train, x_test, y_test)
        return cls._best_alpha_ridge

    @classmethod
    def get_best_accuracy_logistic(cls, x_train, y_train, x_test, y_test):
        if cls._best_accuracy_logistic == 0:
            cls._set_hyperparameters(x_train, y_train, x_test, y_test)
        return cls._best_accuracy_logistic

    @classmethod
    def get_best_alpha_logistic(cls, x_train, y_train, x_test, y_test):
        if cls._best_alpha_logistic == 0:
            cls._set_hyperparameters(x_train, y_train, x_test, y_test)
        return cls._best_alpha_logistic

    @classmethod
    def get_best_accuracy_svm(cls, x_train, y_train, x_test, y_test):
        if cls._best_accuracy_svm == 0:
            cls._set_hyperparameters(x_train, y_train, x_test, y_test)
        return cls._best_accuracy_svm

    @classmethod
    def get_best_c_svm(cls, x_train, y_train, x_test, y_test):
        if cls._best_c_svm == 0:
            cls._set_hyperparameters(x_train, y_train, x_test, y_test)
        return cls._best_c_svm

    @classmethod
    def get_best_kernel_svm(cls, x_train, y_train, x_test, y_test):
        if cls._best_kernel_svm == '':
            cls._set_hyperparameters(x_train, y_train, x_test, y_test)
        return cls._best_kernel_svm
