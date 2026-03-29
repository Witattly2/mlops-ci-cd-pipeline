import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def train_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        
        mlflow.log_metric('accuracy', score)
        mlflow.sklearn.log_model(clf, 'model')
        print(f'Model trained with accuracy: {score}')

if __name__ == '__main__':
    train_model()