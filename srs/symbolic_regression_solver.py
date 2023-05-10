from datetime import datetime
import numpy as np

from srs.result import Result
from srs.evolutionary import EvolutionaryAlg


class SymbolicRegressionSolver:
    def __init__(self, pop_size=100, max_generations=30, max_tree_depth=7, min_tree_depth=2, crossover_rate=0.9, mutation_rate=0.05, elitism_rate=0.1, tournament_size=2):
        self.evolutionary_alg = EvolutionaryAlg(pop_size, 
                                                max_generations, 
                                                max_tree_depth, 
                                                min_tree_depth, 
                                                crossover_rate, 
                                                mutation_rate, 
                                                elitism_rate, 
                                                tournament_size)
        
        self.results = [] # Lista de <Result()>'s

    def fit(self, X, y):
        # Evoluir até o número máximo de gerações a ser alcançado
        self.evolutionary_alg.Evolve(X, y)

    def predict(self, X):
        # Retornar um vetor com as predições para cada linha de X
        pass

    def score(self, y):
        # Comparar as predições com os valores reais e retornar a acurácia
        # Comparar indicadores de Teste x Treino
        pass

    def print_predicted_expression(self):
        print("Not implemented")