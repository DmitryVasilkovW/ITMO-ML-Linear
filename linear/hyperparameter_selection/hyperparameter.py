from linear.classifier.regression import RidgeRegression, LogisticRegressionGD
from linear.classifier.svm_classifier import SVM
from linear.dataset.data import accuracy_metric


def params(X_train, y_train, X_test, y_test):
    best_accuracy_ridge = 0
    best_alpha_ridge = 0

    # alphas_ridge = [0.01, 0.1, 1.0, 10.0]
    alphas_ridge = [0.01, 0.3, 1.0, 10.0]
    for alpha in alphas_ridge:
        ridge = RidgeRegression(alpha=alpha)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict_classes(X_test)
        accuracy = accuracy_metric(y_test, y_pred)
        if accuracy > best_accuracy_ridge:
            best_accuracy_ridge = accuracy
            best_alpha_ridge = alpha

    # Перебор гиперпараметров для Logistic Regression
    best_accuracy_logistic = 0
    best_alpha_logistic = 0

    # alphas_logistic = [0.01, 0.1, 1.0]
    alphas_logistic = [0.01, 0.3, 1.0]
    for alpha in alphas_logistic:
        log_reg = LogisticRegressionGD(alpha=alpha, iterations=1000, penalty='l2')
        log_reg.fit(X_train, y_train)
        y_pred = log_reg.predict_classes(X_test)
        accuracy = accuracy_metric(y_test, y_pred)
        if accuracy > best_accuracy_logistic:
            best_accuracy_logistic = accuracy
            best_alpha_logistic = alpha

    # Перебор гиперпараметров для SVM
    best_accuracy_svm = 0
    best_C_svm = 0
    best_kernel_svm = ''
    kernels = ['linear', 'polynomial', 'rbf']
    Cs = [0.1, 0.3]

    for C in Cs:
        for kernel in kernels:
            svm = SVM(C=C, alpha=0.01, kernel=kernel)
            svm.fit(X_train, 2 * y_train - 1)
            y_pred = svm.predict(X_test)
            accuracy = accuracy_metric(y_test, (y_pred >= 0).astype(int))
            if accuracy > best_accuracy_svm:
                best_accuracy_svm = accuracy
                best_C_svm = C
                best_kernel_svm = kernel

    print(f'Best alpha for Ridge Regression: {best_alpha_ridge}, Best accuracy: {best_accuracy_ridge:.2f}')
    print(f'Best alpha for Logistic Regression: {best_alpha_logistic}, Best accuracy: {best_accuracy_logistic:.2f}')
    print(f'Best C for SVM: {best_C_svm}, Best kernel: {best_kernel_svm}, Best accuracy: {best_accuracy_svm:.2f}')

    return best_accuracy_ridge, best_alpha_ridge, best_accuracy_logistic, best_alpha_logistic, best_C_svm, best_kernel_svm
