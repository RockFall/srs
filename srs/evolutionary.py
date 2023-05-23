from datetime import datetime
#from tqdm import tqdm
import numpy as np
import copy

from srs.cfg import CFG
from srs.operators import crossover, mutate, tournament, roulette
from srs.util import _protected_division


class EvolutionaryAlg:
    def protected_division(x, y):
        if y == 0:
            return 1
        return x / y

    def __init__(self, pop_size=100, max_generations=30, max_tree_depth=7, min_tree_depth=2, crossover_rate=0.9, mutation_rate=0.05, elitism_size=10, tournament_size=2):
        # Loading parameters
        self.params = {}
        self.params["POP_SIZE"] = pop_size
        self.params["MAX_GENERATIONS"] = max_generations
        self.params["MAX_TREE_DEPTH"] = max_tree_depth
        self.params["MIN_TREE_DEPTH"] = min_tree_depth
        self.params["CROSSOVER_RATE"] = crossover_rate
        self.params["MUTATION_RATE"] = mutation_rate
        self.params["ELITISM_SIZE"] = elitism_size
        self.params["TOURNAMENT_SIZE"] = tournament_size
        self.params["LEARNING_FACTOR"] = 0.01
        self.params['INCREMENT_LEARNING_FACTOR'] = 0.0001
        self.params['SEED'] = int(datetime.now().microsecond)
        np.random.seed(int(self.params['SEED']))

        self._invalid_fitness_value = 1e10
        self.best = None
        self.worst = None

        # Defining the grammar used
        grammar_dic = {
            '<start>': [
                [('<expr>', 'NT')]
            ], 
            '<expr>': [
                [('<expr>', 'NT'), ('<op>', 'NT'), ('<expr>', 'NT')], 
                [('(', 'T'), ('<expr>', 'NT'), ('<op>', 'NT'), ('<expr>', 'NT'), (')', 'T')], 
                [('<var>', 'NT')]
            ],
            '<op>': [
                [('+', 'T')], 
                [('-', 'T')], 
                [('*', 'T')], 
                [('|protec_div|', 'T')]
            ],
            '<var>': [
                [('x[0]', 'T')], 
                [('x[1]', 'T')], 
                [('1.0', 'T')]
            ]
        }

        shortest_path = {
            ('<start>', 'NT'): [3, grammar_dic['<start>'][0]],
            ('<expr>', 'NT'): [2, grammar_dic['<expr>'][2]],
            ('<var>', 'NT'): [1, grammar_dic['<var>'][0], grammar_dic['<var>'][1]],
            ('<op>', 'NT'): [1,grammar_dic['<op>'][0], grammar_dic['<op>'][1],grammar_dic['<op>'][2], grammar_dic['<op>'][3]]
        }

        self.cfg = CFG(grammar_dic,
                           max_tree_depth=self.params['MAX_TREE_DEPTH'], 
                           min_tree_depth=self.params['MIN_TREE_DEPTH'],
                           shortest_path=shortest_path)
        

    """
        _____________________________________________________________________________________________
                                 Generation of indidivuals and population
    """
    def generate_random_individual_aux(self, genome, symbol, curr_depth):
        codon = np.random.uniform()
        if curr_depth > self.params['MIN_TREE_DEPTH']:
            prob_non_recursive = 0.0
            for rule in self.cfg.shortest_path[(symbol,'NT')][1:]:
                index = self.cfg.grammar[symbol].index(rule)
                prob_non_recursive += self.cfg.pcfg[self.cfg.index_of_non_terminal[symbol],index]
            prob_aux = 0.0
            for rule in self.cfg.shortest_path[(symbol,'NT')][1:]:
                index = self.cfg.grammar[symbol].index(rule)
                new_prob = self.cfg.pcfg[self.cfg.index_of_non_terminal[symbol],index] / prob_non_recursive
                prob_aux += new_prob
                if codon <= round(prob_aux,3):
                    expansion_possibility = index
                    break
        else:
            prob_aux = 0.0
            for index, option in enumerate(self.cfg.grammar[symbol]):
                prob_aux += self.cfg.pcfg[self.cfg.index_of_non_terminal[symbol],index]
                if codon <= round(prob_aux,3):
                    expansion_possibility = index
                    break
        
        genome[self.cfg.non_terminals.index(symbol)].append([expansion_possibility,codon])
        expansion_symbols = self.cfg.grammar[symbol][expansion_possibility]
        depths = [curr_depth]
        for sym in expansion_symbols:
            if sym[1] != "T":
                depths.append(self.generate_random_individual_aux(genome, sym[0], curr_depth + 1))
        return max(depths)
    
    def generate_random_individual(self):
        genotype = [[] for n in range(len(self.cfg.non_terminals))]
        tree_depth = self.generate_random_individual_aux(genotype, self.cfg.start_rule, 0)
        return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth}

    def generate_random_population(self):
        for _ in range(self.params['POP_SIZE']):
            yield self.generate_random_individual()


    """
        _____________________________________________________________________________________________
                                        Evaluation of an individual
    """
    def Xy_evaluate(self, phenotype, X, y):
        if phenotype is None:
            return None
        
        # For each row in X, calculate the result of the phenotype
        error = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            try:
                result = eval(phenotype, globals(), {"x": X[i], "protec_div": _protected_division})
                error[i] = (y[i] - result)**2
            except (OverflowError, ValueError, ZeroDivisionError) as e:
                return self._invalid_fitness_value
        
        N = X.shape[0]
        fitness_value = np.sqrt( error.sum() / N)

        if fitness_value is None:
            return self._invalid_fitness_value
        
        if np.isnan(fitness_value):
            return self._invalid_fitness_value

        return fitness_value

    def evaluate(self, individual, X, y):
        mapping_values = [0 for _ in individual['genotype']]
        phen, tree_depth = self.cfg.mapping(individual['genotype'], mapping_values)
        quality = self.Xy_evaluate(phen, X, y) # Apply X, y here
        individual['phenotype'] = phen
        individual['fitness'] = quality
        individual['mapping_values'] = mapping_values
        individual['tree_depth'] = tree_depth


    """
        _____________________________________________________________________________________________
                                            PCFG Update
    """
    def prod_rule_expansion_counter(self, genotype):
        # Count the number of times each production rule is used on given genotype
        prod_counter = []
        for nt in self.cfg.non_terminals:
            expansion_list = genotype[self.cfg.non_terminals.index(nt)]
            counter = [0] * len(self.cfg.grammar[nt])
            for prod, _ in expansion_list:
                counter[prod] += 1
            prod_counter.append(counter)
        return prod_counter

    def update_probs(self, best=None):
        if best is None:
            best = self.best

        prod_counter = self.prod_rule_expansion_counter(best['genotype'])
        rows, columns = self.cfg.pcfg.shape
        mask = copy.deepcopy(self.cfg.pcfg_mask)
        for i in range(rows):
            if np.count_nonzero(mask[i,:]) <= 1:
                continue
            total = sum(prod_counter[i])

            for j in range(columns):
                if not mask[i,j]:
                    continue
                counter = prod_counter[i][j]
                old_prob = self.cfg.pcfg[i][j]

                if counter > 0:
                    self.cfg.pcfg[i][j] = min(old_prob + self.params['LEARNING_FACTOR'] * counter / total, 1.0)
                elif counter == 0:
                    self.cfg.pcfg[i][j] = max(old_prob - self.params['LEARNING_FACTOR'] * old_prob, 0.0)

            self.cfg.pcfg[i,:] = np.clip(self.cfg.pcfg[i,:], 0, np.infty) / np.sum(np.clip(self.cfg.pcfg[i,:], 0, np.infty))
        

    """
        _____________________________________________________________________________________________
                                                Evolution
    """
    def get_worst_idx(self, pop):
        idx_worst = -1
        while pop[idx_worst]['fitness'] >= self._invalid_fitness_value:
            idx_worst -= 1
        return idx_worst

    def Evolve(self, X, y):
        # Initial pop
        pop = list(self.generate_random_population())

        # Setup
        use_best_of_gen = False
        best_of_gen = None
        it = 0

        # Evaluate initial pop
        for i in pop:
            if i['fitness'] is None:
                self.evaluate(i, X, y)

        # Start evolution
        data = []
        print("Starting evolution...\nGeneration: ")
        while it <= self.params['MAX_GENERATIONS']:
            print(it, end=' ')
            pop.sort(key=lambda x: x['fitness'])

            # Save best and worst
            if self.best is None:
                self.best = copy.deepcopy(pop[0])
                idx_worst = self.get_worst_idx(pop)
                self.worst = copy.deepcopy(pop[idx_worst])
            else:
                if self.best['fitness'] > pop[0]['fitness']:
                    self.best = copy.deepcopy(pop[0])

                idx_worst = self.get_worst_idx(pop)
                if self.worst['fitness'] < pop[idx_worst]['fitness']:
                    self.worst = copy.deepcopy(pop[idx_worst])

            # Alternate between best overall and best of generation
            if use_best_of_gen:
                self.update_probs()
            else:
                self.update_probs(best_of_gen)
            use_best_of_gen = not use_best_of_gen

            # Increment learning factor
            if self.params['INCREMENT_LEARNING_FACTOR'] > 0:
                self.params['LEARNING_FACTOR'] += self.params['INCREMENT_LEARNING_FACTOR']
        
            # Log here: (it, pop, best, grammar.get_pcfg())

            # Generate new population
            new_pop = []
            crossover_improvement_count = 0
            crossover_degradation_count = 0
            crossover_total_count = 0
            while len(new_pop) < self.params['POP_SIZE'] - self.params['ELITISM_SIZE']:
                # Tournament selection
                if self.params['TOURNAMENT_SIZE'] > 0:
                    if np.random.uniform() < self.params['CROSSOVER_RATE']:
                        # WITH crossover
                        crossover_total_count += 1
                        non_inite_loop_count = 10000
                        while True:
                            parent1 = tournament(pop, self.params['TOURNAMENT_SIZE'])
                            parent2 = tournament(pop, self.params['TOURNAMENT_SIZE'])
                            new_individual = crossover(parent1, parent2, self.cfg)
                            self.evaluate(new_individual, X, y)
                            non_inite_loop_count -= 1
                            if new_individual['fitness'] < self._invalid_fitness_value or non_inite_loop_count == 0:
                                break

                        mean_parents_fitness = (parent1['fitness'] + parent2['fitness']) / 2

                        if new_individual['fitness'] > mean_parents_fitness:
                            crossover_improvement_count += 1
                        elif new_individual['fitness'] < mean_parents_fitness:
                            crossover_degradation_count += 1
                    else:
                        # WITHOUT crossover
                        new_individual = tournament(pop, self.params['TOURNAMENT_SIZE'])

                # Roulllete selection
                else:
                    if np.random.uniform() < self.params['CROSSOVER_RATE']:
                        # WITH crossover
                        crossover_total_count += 1
                        non_inite_loop_count = 10000
                        while True:
                            parent1 = roulette(pop)
                            parent2 = roulette(pop)
                            new_individual = crossover(parent1, parent2)
                            self.evaluate(new_individual, X, y)
                            non_inite_loop_count -= 1
                            if new_individual['fitness'] < self._invalid_fitness_value or non_inite_loop_count == 0:
                                break

                        mean_parents_fitness = (parent1['fitness'] + parent2['fitness']) / 2

                        if new_individual['fitness'] > mean_parents_fitness:
                            crossover_improvement_count += 1
                        elif new_individual['fitness'] < mean_parents_fitness:
                            crossover_degradation_count += 1
                    else:
                        # WITHOUT crossover
                        new_individual = roulette(pop)

                # Mutation
                non_inite_loop_count = 10000
                while True:
                    new_individual = mutate(new_individual, grammar=self.cfg, pmutation=self.params['MUTATION_RATE'])
                    self.evaluate(new_individual, X, y)
                    non_inite_loop_count -= 1
                    if new_individual['fitness'] < self._invalid_fitness_value or non_inite_loop_count == 0:
                        break

                new_pop.append(new_individual)

            # Evaluate new population
            for i in new_pop:
                if i['fitness'] is None:
                    self.evaluate(i, X, y)
            new_pop.sort(key=lambda x: x['fitness'])

            # best individual from the current generation
            best_of_gen = copy.deepcopy(new_pop[0])

            for i in pop[:self.params['ELITISM_SIZE']]:
                self.evaluate(i, X, y)
            new_pop += pop[:self.params['ELITISM_SIZE']]

            idx_worst = self.get_worst_idx(new_pop)
            generation_data = {'iteration': it,
                               'best_all': self.best,
                               'best_all_fitness': self.best['fitness'],
                               'best_curr': best_of_gen,
                               'best_curr_fitness': best_of_gen['fitness'],
                               'worst_all': self.worst,
                               'worst_all_fitness': self.worst['fitness'],
                               'worst_curr': new_pop[idx_worst], 
                               'worst_curr_fitness': new_pop[idx_worst]['fitness'],
                               'avg': np.mean([i['fitness'] for i in pop])
                               }
            generation_data['repeated_count'] = EvolutionaryAlg.find_repeated_individuals_count(pop)
            generation_data['crossover_total_count'] = crossover_total_count
            generation_data['crossover_improved'] = crossover_improvement_count
            generation_data['crossover_degraded'] = crossover_degradation_count
            data.append(generation_data)
            pop = new_pop
            it += 1


        return data
    
    """
        _____________________________________________________________________________________________
                                                Utils
    """
    def find_repeated_individuals_count(pop):
        unique_phenos = set()
        repeated_count = 0
        for i in pop:
            pheno = i['phenotype']
            if pheno in unique_phenos:
                repeated_count += 1
            else:
                unique_phenos
        return repeated_count
        
        