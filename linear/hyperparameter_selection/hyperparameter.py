from linear.classifier.regression import RidgeRegression, LogisticRegressionGD
from linear.classifier.svm_classifier import SVM
from linear.dataset.data import accuracy_metric


class HyperparameterSelection:
    best_accuracy_ridge = 0
    best_alpha_ridge = 0
    best_accuracy_logistic = 0
    best_alpha_logistic = 0
    best_accuracy_svm = 0
    best_c_svm = 0
    best_kernel_svm = ''

    @classmethod
    def _set_hyperparameters(cls, x_train, y_train, x_test, y_test):
        cls._clean()
        alphas_ridge = [0.01, 0.3, 1.0]
        for alpha in alphas_ridge:
            ridge = RidgeRegression(alpha=alpha)
            ridge.fit(x_train, y_train)
            y_pred = ridge.predict_classes(x_test)
            accuracy = accuracy_metric(y_test, y_pred)
            if accuracy > cls.best_accuracy_ridge:
                cls.best_accuracy_ridge = accuracy
                cls.best_alpha_ridge = alpha

        alphas_logistic = [0.01, 1.0]
        for alpha in alphas_logistic:
            log_reg = LogisticRegressionGD(alpha=alpha, iterations=1000, penalty='l2')
            log_reg.fit(x_train, y_train)
            y_pred = log_reg.predict_classes(x_test)
            accuracy = accuracy_metric(y_test, y_pred)
            if accuracy > cls.best_accuracy_logistic:
                cls.best_accuracy_logistic = accuracy
                cls.best_alpha_logistic = alpha

        kernels = ['linear', 'polynomial', 'rbf']
        cs = [0.1, 0.3]

        for C in cs:
            for kernel in kernels:
                svm = SVM(C=C, alpha=0.01, kernel=kernel)
                svm.fit(x_train, 2 * y_train - 1)
                y_pred = svm.predict(x_test)
                accuracy = accuracy_metric(y_test, (y_pred >= 0).astype(int))
                if accuracy > cls.best_accuracy_svm:
                    cls.best_accuracy_svm = accuracy
                    cls.best_c_svm = C
                    cls.best_kernel_svm = kernel

        print(
            f'Best alpha for Ridge Regression: {cls.best_alpha_ridge},'
            f' Best accuracy: {cls.best_accuracy_ridge:.2f}')
        print(
            f'Best alpha for Logistic Regression: {cls.best_alpha_logistic},'
            f' Best accuracy: {cls.best_accuracy_logistic:.2f}')
        print(
            f'Best C for SVM: {cls.best_c_svm}, Best kernel: {cls.best_kernel_svm},'
            f' Best accuracy: {cls.best_accuracy_svm:.2f}')

    @classmethod
    def _clean(cls):
        cls.best_accuracy_ridge = 0
        cls.best_alpha_ridge = 0
        cls.best_accuracy_logistic = 0
        cls.best_alpha_logistic = 0
        cls.best_accuracy_svm = 0
        cls.best_c_svm = 0
        cls.best_kernel_svm = ''

    @classmethod
    def get_best_accuracy_ridge(cls, x_train, y_train, x_test, y_test):
        if cls.best_accuracy_ridge == 0:
            cls._set_hyperparameters(x_train, y_train, x_test, y_test)
        return cls.best_accuracy_ridge

    @classmethod
    def get_best_alpha_ridge(cls, x_train, y_train, x_test, y_test):
        if cls.best_alpha_ridge == 0:
            cls._set_hyperparameters(x_train, y_train, x_test, y_test)
        return cls.best_alpha_ridge

    @classmethod
    def get_best_accuracy_logistic(cls, x_train, y_train, x_test, y_test):
        if cls.best_accuracy_logistic == 0:
            cls._set_hyperparameters(x_train, y_train, x_test, y_test)
        return cls.best_accuracy_logistic

    @classmethod
    def get_best_alpha_logistic(cls, x_train, y_train, x_test, y_test):
        if cls.best_alpha_logistic == 0:
            cls._set_hyperparameters(x_train, y_train, x_test, y_test)
        return cls.best_alpha_logistic

    @classmethod
    def get_best_accuracy_svm(cls, x_train, y_train, x_test, y_test):
        if cls.best_accuracy_svm == 0:
            cls._set_hyperparameters(x_train, y_train, x_test, y_test)
        return cls.best_accuracy_svm

    @classmethod
    def get_best_c_svm(cls, x_train, y_train, x_test, y_test):
        if cls.best_c_svm == 0:
            cls._set_hyperparameters(x_train, y_train, x_test, y_test)
        return cls.best_c_svm

    @classmethod
    def get_best_kernel_svm(cls, x_train, y_train, x_test, y_test):
        if cls.best_kernel_svm == '':
            cls._set_hyperparameters(x_train, y_train, x_test, y_test)
        return cls.best_kernel_svm
