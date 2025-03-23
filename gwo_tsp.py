"""
ENHANCED GREY WOLF OPTIMIZER FOR TSP
=====================================
Advanced implementation with diversity preservation and adaptive operators.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

class EnhancedTSPGWO:
    """Advanced GWO implementation with multiple improvement strategies."""
    
    DEFAULT_CONFIG = {
        'population_size': 100,
        'max_iterations': 500,
        'a_decay_rate': 2.0,
        'initial_mutation_rate': 0.3,
        'final_mutation_rate': 0.1,
        'diversity_threshold': 0.5,
        'log_level': logging.INFO,
        'output_dir': 'enhanced_gwo_results',
        'random_seed': None
    }

    def __init__(self, distance_matrix: np.ndarray, config: Optional[dict] = None):
        self._validate_matrix(distance_matrix)
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self._prepare_output_dir()
        self._setup_logging()
        
        self.rng = np.random.default_rng(self.config['random_seed'])
        self.population = []
        self.fitness = []
        self.alpha = {'solution': None, 'fitness': float('inf')}
        self.beta = {'solution': None, 'fitness': float('inf')}
        self.delta = {'solution': None, 'fitness': float('inf')}
        self.convergence = []
        self.execution_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    def _validate_matrix(self, matrix):
        """Validate TSP distance matrix."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Distance matrix must be square")
        if np.any(matrix < 0):
            raise ValueError("Negative distances not allowed")
        np.fill_diagonal(matrix, 0)

    def _prepare_output_dir(self):
        """Ensure output directory exists."""
        os.makedirs(self.config['output_dir'], exist_ok=True)

    def _setup_logging(self):
        """Configure logging system."""
        logging.basicConfig(
            level=self.config['log_level'],
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config['output_dir']}/execution.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EnhancedGWO')
        self.logger.info("Initialized solver for %d cities", self.num_cities)

    def _initialize_population(self):
        """Generate diverse initial population using multiple strategies."""
        self.population = []
        
        # 50% random permutations
        for _ in range(self.config['population_size'] // 2):
            self.population.append(self.rng.permutation(self.num_cities).tolist())
        
        # 25% nearest neighbor starts
        for _ in range(self.config['population_size'] // 4):
            start = self.rng.integers(self.num_cities)
            tour = [start]
            current = start
            while len(tour) < self.num_cities:
                next_city = np.argmin([self.distance_matrix[current, j] if j not in tour else np.inf 
                                      for j in range(self.num_cities)])
                tour.append(next_city)
                current = next_city
            self.population.append(tour)
        
        # 25% greedy insertion
        for _ in range(self.config['population_size'] - len(self.population)):
            tour = [self.rng.integers(self.num_cities)]
            while len(tour) < self.num_cities:
                best_cost = np.inf
                best_pos = -1
                best_city = -1
                for city in range(self.num_cities):
                    if city not in tour:
                        for pos in range(len(tour)+1):
                            new_tour = tour[:pos] + [city] + tour[pos:]
                            cost = self._evaluate_fitness(new_tour)
                            if cost < best_cost:
                                best_cost = cost
                                best_pos = pos
                                best_city = city
                tour.insert(best_pos, best_city)
            self.population.append(tour)

    def _evaluate_fitness(self, solution: List[int]) -> float:
        """Calculate total tour distance."""
        return sum(
            self.distance_matrix[solution[i], solution[(i+1)%len(solution)]]
            for i in range(len(solution))
        )

    def _update_hierarchy(self):
        """Update alpha, beta, delta leaders."""
        sorted_indices = np.argsort(self.fitness)
        self.alpha = {
            'solution': self.population[sorted_indices[0]],
            'fitness': self.fitness[sorted_indices[0]]
        }
        self.beta = {
            'solution': self.population[sorted_indices[1]],
            'fitness': self.fitness[sorted_indices[1]]
        }
        self.delta = {
            'solution': self.population[sorted_indices[2]],
            'fitness': self.fitness[sorted_indices[2]]
        }

    def _edge_recombination_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Edge recombination crossover operator."""
        edge_map = defaultdict(set)
        for tour in [parent1, parent2]:
            for i in range(len(tour)):
                current = tour[i]
                prev = tour[(i-1)%len(tour)]
                next_city = tour[(i+1)%len(tour)]
                edge_map[current].add(prev)
                edge_map[current].add(next_city)
        
        child = []
        current = self.rng.choice(parent1)
        while len(child) < self.num_cities:
            child.append(current)
            neighbors = []
            for n in edge_map[current]:
                if n not in child:
                    neighbors.append(n)
            if neighbors:
                current = min(neighbors, key=lambda x: len(edge_map[x]))
            else:
                remaining = [city for city in parent1 if city not in child]
                current = self.rng.choice(remaining) if remaining else None
        return child

    def _adaptive_mutation(self, solution: List[int], iteration: int) -> List[int]:
        """Apply adaptive mutation with multiple operators."""
        mutation_rate = self.config['initial_mutation_rate'] - \
                       (self.config['initial_mutation_rate'] - self.config['final_mutation_rate']) * \
                       (iteration / self.config['max_iterations'])
        
        if self.rng.random() < mutation_rate:
            mutation_type = self.rng.choice(['swap', 'inversion', 'scramble'])
            
            if mutation_type == 'swap':
                i, j = self.rng.choice(self.num_cities, 2, replace=False)
                solution[i], solution[j] = solution[j], solution[i]
                
            elif mutation_type == 'inversion':
                start, end = sorted(self.rng.choice(self.num_cities, 2, replace=False))
                solution[start:end+1] = solution[start:end+1][::-1]
                
            elif mutation_type == 'scramble':
                start, end = sorted(self.rng.choice(self.num_cities, 2, replace=False))
                segment = solution[start:end+1]
                self.rng.shuffle(segment)
                solution[start:end+1] = segment
                
        return solution

    def optimize(self) -> Dict:
        """Enhanced optimization process with diversity control."""
        self._initialize_population()
        self.fitness = [self._evaluate_fitness(ind) for ind in self.population]
        self._update_hierarchy()
        self.convergence = [self.alpha['fitness']]
        
        for iteration in range(self.config['max_iterations']):
            # Adaptive parameter calculation
            a = self.config['a_decay_rate'] * (1 - iteration/self.config['max_iterations'])
            
            new_population = []
            for wolf in self.population:
                # Generate candidate solutions using multiple strategies
                if self.rng.random() < 0.5:
                    child_alpha = self._edge_recombination_crossover(wolf, self.alpha['solution'])
                    child_beta = self._edge_recombination_crossover(wolf, self.beta['solution'])
                    child_delta = self._edge_recombination_crossover(wolf, self.delta['solution'])
                else:
                    child_alpha = self._edge_recombination_crossover(self.alpha['solution'], self.beta['solution'])
                    child_beta = self._edge_recombination_crossover(self.beta['solution'], self.delta['solution'])
                    child_delta = self._edge_recombination_crossover(self.delta['solution'], self.alpha['solution'])
                
                # Select and mutate best candidate
                candidates = [child_alpha, child_beta, child_delta]
                new_wolf = self._adaptive_mutation(
                    min(candidates, key=lambda x: self._evaluate_fitness(x)),
                    iteration
                )
                
                # Prevent duplicate solutions
                if new_wolf not in new_population:
                    new_population.append(new_wolf)
                else:
                    new_population.append(self.rng.permutation(self.num_cities).tolist())
            
            self.population = new_population
            self.fitness = [self._evaluate_fitness(ind) for ind in self.population]
            self._update_hierarchy()
            self.convergence.append(self.alpha['fitness'])
            
            # Diversity preservation
            unique_solutions = len(set(map(tuple, self.population)))
            if unique_solutions < self.config['population_size'] * self.config['diversity_threshold']:
                num_new = int(self.config['population_size'] * 0.25)
                new_solutions = [self.rng.permutation(self.num_cities).tolist() 
                                for _ in range(num_new)]
                self.population[-num_new:] = new_solutions
                self.fitness[-num_new:] = [self._evaluate_fitness(ind) for ind in new_solutions]
                self._update_hierarchy()
            
            # Progress logging
            self.logger.info(
                "Iter %d/%d: Best=%.2f Diversity=%d/%d",
                iteration+1, self.config['max_iterations'],
                self.alpha['fitness'],
                unique_solutions,
                self.config['population_size']
            )
        
        self._generate_outputs()
        return self.alpha

    def _generate_outputs(self):
        """Generate optimization reports and visualizations."""
        # Convergence plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence, 'b-', linewidth=2)
        plt.title('Optimization Progress', fontsize=14)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Best Tour Distance', fontsize=12)
        plt.grid(True, alpha=0.3)
        plot_path = f"{self.config['output_dir']}/convergence_{self.execution_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Text report
        report = f"""
        ENHANCED GWO OPTIMIZATION REPORT
        ================================
        - Number of Cities: {self.num_cities}
        - Best Route Distance: {self.alpha['fitness']:.2f}
        - Optimal Route Sequence: {' → '.join(map(str, self.alpha['solution']))}
        - Initial Distance: {self.convergence[0]:.2f}
        - Final Improvement: {((self.convergence[0] - self.alpha['fitness'])/self.convergence[0]*100):.1f}%
        - Total Iterations: {len(self.convergence)}
        - Final Diversity: {len(set(map(tuple, self.population)))}/{self.config['population_size']}
        - Convergence Plot: {plot_path}
        """
        print(report)
        with open(f"{self.config['output_dir']}/report_{self.execution_id}.txt", 'w') as f:
            f.write(report)

if __name__ == "__main__":
    # Example 5-city problem
    dm = np.array([
        [0, 10, 15, 20, 25],
        [10, 0, 35, 25, 30],
        [15, 35, 0, 30, 40],
        [20, 25, 30, 0, 45],
        [25, 30, 40, 45, 0]
    ])
    
    solver = EnhancedTSPGWO(dm, {
        'population_size': 150,
        'max_iterations': 1000,
        'output_dir': 'enhanced_gwo_demo'
    })
    result = solver.optimize()
    
    print(f"\nOptimal Route: {result['solution']}")
    print(f"Total Distance: {result['fitness']:.2f}")