from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def test_model_training():
    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_model_accuracy():
    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    assert accuracy >= 0.80
