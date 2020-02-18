def evaluate_model_performance(clf,
                               X_train,
                               y_train,
                               class_name):
    """ Prints a model's accuracy and F1-score

    INPUT:
        clf: Model object

        X_train: Training data matrix

        y_train: Expected model output vector

        class_name: Name of the class used for printing purpose

    OUTPUT:
        clf_accuracy: Model accuracy

        clf_f1_score: Model F1-score"""

    y_pred_rf = clf.predict(X_train)

    clf_accuracy = accuracy_score(y_train, y_pred_rf)
    clf_f1_score = f1_score(y_train, y_pred_rf)
    clf_Precision = precision_score(y_train, y_pred_rf)
    clf_Recall = recall_score(y_train, y_pred_rf)

    print("%s model accuracy: %.3f" % (class_name, clf_accuracy))
    print("%s model f1-score: %.3f" % (class_name, clf_f1_score))
    print("%s model Precision: %.3f" % (class_name, clf_Precision))
    print("%s model Recall: %.3f" % (class_name, clf_Recall))

    return clf_accuracy, clf_f1_score, clf_Precision, clf_Recall
