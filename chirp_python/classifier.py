import numpy as np
from chirp_python.trainer import Trainer
from chirp_python.scorer import Scorer

class Classifier:
    debug = False

    def __init__(self, n_reps, train_data, seed, m_debug):
        self.n_reps = n_reps
        self.training_data = train_data
        self.seed = seed
        Classifier.debug = m_debug
        self.trainer = [None] * n_reps
        self.n_classes = 0

    def build_classifier(self):
        random = np.random.RandomState(self.seed)
        self.n_classes = len(np.unique(self.training_data.iloc[:, -1]))
        
        for rep in range(self.n_reps):
            if Classifier.debug:
                print(f"\nVoter: {rep + 1}")
            
            if self.training_data.shape[1] <= 1:
                return
                
            self.trainer[rep] = Trainer(self.training_data, random, self.n_reps)
            self.trainer[rep].classify()

    def classify_instance(self, instance):
        Scorer.voter = 0
        counts = np.zeros(self.n_classes, dtype=int)
        
        for rep in range(self.n_reps):
            Scorer.voter += 1
            if self.trainer[rep] is None:
                return 0
                
            result = self.trainer[rep].score(instance, rep)
            
            if result is None:
                return 0
                
            predicted, weight = result
            counts[predicted] += weight
            
        decision = np.argmax(counts)
        return decision
