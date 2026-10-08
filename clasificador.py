"""Clasificador Naive Bayes multinomial para vectores de documentos."""
import math

class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
        self.clases = set()
        self.vocab_size = 0

    def entrenar(self, X, y):
        if not X or not y or len(X) != len(y) or not X[0]:
            raise ValueError("Se necesitan documentos etiquetados y un vocabulario no vacío.")
        self.clases = set(y)
        self.vocab_size = len(X[0])
        total_docs = len(y)
        conteo_clases = {}
        conteo_palabras = {}

        for i in range(total_docs):
            clase = y[i]
            if clase not in conteo_clases:
                conteo_clases[clase] = 0
                conteo_palabras[clase] = [0] * self.vocab_size
            conteo_clases[clase] += 1
            for j in range(self.vocab_size):
                conteo_palabras[clase][j] += X[i][j]

        self.priors = {c: conteo_clases[c]/total_docs for c in self.clases}
        self.likelihoods = {}
        for c in self.clases:
            total_palabras = sum(conteo_palabras[c])
            self.likelihoods[c] = [
                (conteo_palabras[c][i] + 1) / (total_palabras + self.vocab_size)
                for i in range(self.vocab_size)
            ]

    def predict(self, X):
        return [self._predecir_una(x) for x in X]

    def _predecir_una(self, x):
        log_scores = self.predict_logScores([x])[0]
        return max(log_scores, key=log_scores.get)

    def predict_probabilities(self, X):
        resultados = []
        for scores in self.predict_logScores(X):
            maximum = max(scores.values())
            weights = {c: 2 ** (value - maximum) for c, value in scores.items()}
            total = sum(weights.values())
            resultados.append({c: value / total for c, value in weights.items()})
        return resultados

    def predict_logScores(self, X):
        resultados = []
        for x in X:
            log_scores = {}
            for c in self.clases:
                log_prob = math.log2(self.priors[c])
                for i in range(self.vocab_size):
                    if x[i] > 0:
                        log_prob += x[i] * math.log2(self.likelihoods[c][i])
                log_scores[c] = log_prob
            resultados.append(log_scores)
        return resultados

