from pyswarm import pso
import numpy as np
from utils import evaluate_feature_subset


def particle_swarm_optimization(x_train, y_train, num_features, num_particles, num_iterations):
    # Define the fitness function for PSO
    def fitness_function(solution):
        selected_features = solution > 0.5  # Convert PSO solution to a binary array for feature selection
        if not np.any(selected_features):
            return float('inf')  # Avoid having no features selected

        # Evaluate the feature subset (you need to implement this function)
        return -evaluate_feature_subset(x_train, y_train, selected_features)

    # Set the bounds for the PSO problem
    lower_bounds = [0] * num_features
    upper_bounds = [1] * num_features

    # Run PSO
    optimal_solution, _ = pso(fitness_function, lower_bounds, upper_bounds, swarmsize=num_particles,
                              maxiter=num_iterations)

    # Convert the PSO solution to a binary array for feature selection
    best_feature_subset = optimal_solution > 0.5
    return best_feature_subset
