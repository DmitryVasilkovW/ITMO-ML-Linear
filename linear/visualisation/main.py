from linear.classifier.regression import LinearRegressionRidge
from linear.classifier.svm_classifier import SVMClassifier
from linear.dataset.data import load_and_preprocess_data
from linear.utils.losses_utils import ridge_loss, logistic_loss, tanh_loss
from linear.utils.training_utils import plot_learning_curves

# URL набора данных
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"


def main():
    # 1. Загрузка и подготовка данных
    X_train, X_test, y_train, y_test = load_and_preprocess_data(url)

    # 2. Линейная регрессия с гребневой регуляризацией
    ridge_model = LinearRegressionRidge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    print("Ridge Loss on Test Data:", ridge_loss(y_test, y_pred_ridge))

    # 3. Метод опорных векторов с разными ядрами
    svm_linear = SVMClassifier(kernel="linear", C=1.0, learning_rate=0.001, epochs=50)
    plot_learning_curves(svm_linear, X_train, y_train, X_test, y_test, logistic_loss,
                         title="SVM Linear Kernel - Logistic Loss")

    svm_poly = SVMClassifier(kernel="polynomial", C=1.0, learning_rate=0.001, epochs=50)
    plot_learning_curves(svm_poly, X_train, y_train, X_test, y_test, tanh_loss,
                         title="SVM Polynomial Kernel - Tanh Loss")

    svm_rbf = SVMClassifier(kernel="rbf", C=1.0, learning_rate=0.001, epochs=50)
    plot_learning_curves(svm_rbf, X_train, y_train, X_test, y_test, logistic_loss,
                         title="SVM RBF Kernel - Logistic Loss")


if __name__ == "__main__":
    main()
