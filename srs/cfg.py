import numpy as np

class CFG:
    def __init__(self, grammar, max_tree_depth=None, min_tree_depth=None, shortest_path=None):
        self.grammar = grammar
        self.max_tree_depth = max_tree_depth
        self.min_tree_depth = min_tree_depth
        self.shortest_path = shortest_path

        self.non_terminals = list(self.grammar.keys())
        self.terminals = grammar['<op>'] + grammar['<var>']
        self.start_rule = '<start>'

        self.index_of_non_terminal = {}
        self.gen_pcfg()

    def gen_pcfg(self):
        """
        assigns uniform probabilities to grammar
        """
        array = np.zeros(shape=(len(self.grammar.keys()),4))
        for i, nt in enumerate(self.grammar):
            number_probs = len(self.grammar[nt])
            prob = 1.0 / number_probs
            array[i,:number_probs] = prob
            if nt not in self.index_of_non_terminal:
                self.index_of_non_terminal[nt] = i
        self.pcfg = array
        self.pcfg_mask = self.pcfg != 0
