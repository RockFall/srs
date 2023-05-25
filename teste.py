import numpy as np
import srs

# Load the datasets
def load_data():
    import os
    
    # Find all CSV files in the datasets folders
    csv_files = []
    for root, dirs, files in os.walk("datasets"):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    # Load the CSV files as numpy matrices
    datasets = {}
    for file in csv_files:
        matrix = np.genfromtxt(file, delimiter=",")
        file_name = os.path.basename(file).split('.')[0]
        datasets[file_name] = matrix

    return datasets

datasets = load_data()

name = 'concrete'
X_train = datasets[name + '-train'][:, :-1]
y_train = datasets[name + '-train'][:, -1]

X_test = datasets[name + '-test'][:, :-1]
y_test = datasets[name + '-test'][:, -1]

solver = srs.SymbolicRegressionSolver()

# Experiments
scores = {}
NUM_EXPERIMENTS_PER_CONFIG = 1
POPULATIONS = [50, 100, 500]
GENERATIONS = [50, 100, 500]
best_of_each = {} # Map {'experiment' : 'name_of_best'}


solver = srs.SymbolicRegressionSolver()

experiment_name = name+'-default-' + '-pop:' + str(30) + '-gens:' + str(50) + '-it:' + str(0)
solver.update_params(pop_size=50, max_generations=100)
solver.fit(X_train, y_train, experiment_name)

print(solver.get_best_param_per_generation('best_all_fitness')[-1])

#y_hat = solver.predict(X_test, name=experiment_name)
#scores[experiment_name] = solver.score(y_test, y_hat)

