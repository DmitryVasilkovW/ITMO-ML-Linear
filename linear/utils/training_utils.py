import matplotlib.pyplot as plt


def plot_learning_curves(model, X_train, y_train, X_test, y_test, loss_func, title="Learning Curve"):
    train_losses = []
    test_losses = []
    epochs = model.epochs

    for epoch in range(1, epochs + 1):
        model.epochs = epoch
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_loss = loss_func(y_train, y_train_pred)
        test_loss = loss_func(y_test, y_test_pred)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", color="blue")
    plt.plot(range(1, epochs + 1), test_losses, label="Test Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()
