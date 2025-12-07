import numpy as np
import pandas as pd
from chirp_python.classifier import Classifier

class CHIRP:
    def __init__(self, num_voters=7, random_seed=1, debug=False):
        self.num_voters = num_voters
        self.random_seed = random_seed
        self.debug = debug
        self.classifier = None

    def fit(self, X, y):
        # In WEKA, it seems the data is passed as a single dataframe with class at the end
        train_data = pd.concat([X, y], axis=1)
        
        self.classifier = Classifier(self.num_voters, train_data, self.random_seed, self.debug)
        self.classifier.build_classifier()

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            instance = X.iloc[i, :].values
            predictions.append(self.classifier.classify_instance(instance))
        return np.array(predictions)

if __name__ == '__main__':
    # Example Usage
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Create a dummy dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y, name='class')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the classifier
    chirp = CHIRP(num_voters=7, random_seed=1, debug=True)
    chirp.fit(X_train, y_train)

    # Make predictions
    y_pred = chirp.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy}")
