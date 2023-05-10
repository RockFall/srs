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
        matrix = np.loadtxt(file, delimiter=",")
        file_name = os.path.basename(file).split('.')[0]
        datasets[file_name] = matrix

    return datasets



solver = srs.SymbolicRegressionSolver()
X_train = np.array([[1, 1], [2, 2], [1, 2]])
y_train = np.array([2, 4, 3])

X_test = np.array([[3, 3], [3, 1], [1, 3]])
y_test = np.array([6, 4, 4])

solver.fit(X_train, y_train)
solver.predict(X_test)
solver.score(y_test)

solver.print_predicted_expression()

# Testing
"""
datasets = load_data()
X_train = datasets['synth1-train'][:, :2]
y_train = datasets['synth1-train'][:, 2]

X_test = datasets['synth1-test'][:, :2]
y_test = datasets['synth1-test'][:, 2]

solver.fit(X_train, y_train)
solver.predict(X_test)
solver.score(y_test)


solver.print_predicted_expression()
print(solver.best_fitness())
print(solver.worst_fitness())
"""