from datetime import datetime
#from tqdm import tqdm
import numpy as np

from srs.cfg import CFG

class EvolutionaryAlg:
    def __init__(self, pop_size=100, max_generations=30, max_tree_depth=7, min_tree_depth=2, crossover_rate=0.9, mutation_rate=0.05, elitism_rate=0.1, tournament_size=2):
        # Loading parameters
        self.params = {}
        self.params["POP_SIZE"] = pop_size
        self.params["MAX_GENERATIONS"] = max_generations
        self.params["MAX_TREE_DEPTH"] = max_tree_depth
        self.params["MIN_TREE_DEPTH"] = min_tree_depth
        self.params["CROSSOVER_RATE"] = crossover_rate
        self.params["MUTATION_RATE"] = mutation_rate
        self.params["ELITISM_RATE"] = elitism_rate
        self.params["TOURNAMENT_SIZE"] = tournament_size
        self.params['SEED'] = int(datetime.now().microsecond)
        np.random.seed(int(self.params['SEED']))

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
                [('\\eb_div_\\eb', 'T')]
            ],
            '<var>': [
                [('x[0]', 'T')], 
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
        for individual in range(self.params['POP_SIZE']):
            yield self.generate_random_individual()

    def evaluate(self, ind, eval_func):
        mapping_values = [0 for _ in ind['genotype']]
        phen, tree_depth = self.cfg.mapping(ind['genotype'], mapping_values)
        quality, other_info = eval_func.evaluate(phen)
        ind['phenotype'] = phen
        ind['fitness'] = quality
        ind['other_info'] = other_info
        ind['mapping_values'] = mapping_values
        ind['tree_depth'] = tree_depth

    def Evolve(self, X, y):
        # Initial pop
        pop = list(self.generate_random_population())
        flag = False    # alternate False - best overall
        best = None
        it = 0
        for i in population:
            if i['fitness'] is None:
                self.evaluate(i, X, y)
        while it <= params['GENERATIONS']:        

            population.sort(key=lambda x: x['fitness'])
            # best individual overall
            if not best:
                best = copy.deepcopy(population[0])
            elif population[0]['fitness'] <= best['fitness']:
                best = copy.deepcopy(population[0])
        
            if not flag:
                update_probs(best, params['LEARNING_FACTOR'])
            else:
                update_probs(best_gen, params['LEARNING_FACTOR'])
            flag = not flag

            if params['ADAPTIVE']:
                params['LEARNING_FACTOR'] += params['ADAPTIVE_INCREMENT']

        
            logger.evolution_progress(it, population, best, grammar.get_pcfg())

            new_population = []
            while len(new_population) < params['POPSIZE'] - params['ELITISM']:
                if np.random.uniform() < params['PROB_CROSSOVER']:
                    p1 = tournament(population, params['TSIZE'])
                    p2 = tournament(population, params['TSIZE'])
                    ni = crossover(p1, p2)
                else:
                    ni = tournament(population, params['TSIZE'])
                ni = mutate(ni, params['PROB_MUTATION'])
                new_population.append(ni)

            for i in tqdm(new_population):
                evaluate(i, evaluation_function)
            new_population.sort(key=lambda x: x['fitness'])
            # best individual from the current generation
            best_gen = copy.deepcopy(new_population[0])

            for i in tqdm(population[:params['ELITISM']]):
                evaluate(i, evaluation_function)
            new_population += population[:params['ELITISM']]

            population = new_population
            it += 1

        