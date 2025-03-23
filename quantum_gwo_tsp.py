"""
QUANTUM-INSPIRED GREY WOLF OPTIMIZER FOR TSP
=============================================
Hybrid quantum-classical implementation with qubit representation.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

class QuantumGWO_TSP:
    """Quantum-inspired Grey Wolf Optimizer with qubit representation."""
    
    DEFAULT_CONFIG = {
        'population_size': 50,
        'max_iterations': 200,
        'quantum_bits': 8,
        'rotation_step': 0.05*np.pi,
        'entanglement_prob': 0.25,
        'decay_rate': 0.98,
        'log_level': logging.INFO,
        'output_dir': 'qgwo_tsp_results',
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
        self.quantum_population = []
        self.classical_population = []
        self.fitness = []
        self.alpha = {'solution': None, 'fitness': float('inf')}
        self.convergence = []
        self.execution_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    def _validate_matrix(self, matrix):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Invalid distance matrix")
        np.fill_diagonal(matrix, 0)

    def _prepare_output_dir(self):
        os.makedirs(self.config['output_dir'], exist_ok=True)

    def _setup_logging(self):
        logging.basicConfig(
            level=self.config['log_level'],
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config['output_dir']}/execution.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('QuantumGWO')

    def _initialize_quantum_population(self):
        """Initialize quantum wolves using qubit representation."""
        self.quantum_population = [
            np.random.uniform(0, 2*np.pi, 
            (self.num_cities, self.config['quantum_bits']))
            for _ in range(self.config['population_size'])
        ]

    def _quantum_measure(self, quantum_state: np.ndarray) -> List[int]:
        """Convert quantum state to classical solution."""
        probabilities = (np.sin(quantum_state) ** 2).sum(axis=1)
        return np.argsort(probabilities).tolist()

    def _quantum_rotate(self, source: np.ndarray, target: np.ndarray):
        """Quantum rotation towards target solution."""
        return source + self.config['rotation_step'] * (target - source)

    def _quantum_entangle(self, state: np.ndarray):
        """Apply quantum entanglement operation."""
        if self.rng.random() < self.config['entanglement_prob']:
            i, j = self.rng.choice(self.num_cities, 2, replace=False)
            state[[i, j]] = state[[j, i]]  # Swap entangled qubits
        return state

    def _calculate_distance(self, route: List[int]) -> float:
        return sum(
            self.distance_matrix[route[i], route[(i+1)%len(route)]]
            for i in range(len(route))
        )

    def optimize(self) -> Dict:
        """Quantum-classical hybrid optimization."""
        self._initialize_quantum_population()
        
        # Initial measurement and evaluation
        self.classical_population = [
            self._quantum_measure(q_state) 
            for q_state in self.quantum_population
        ]
        self.fitness = [self._calculate_distance(r) for r in self.classical_population]
        best_idx = np.argmin(self.fitness)
        self.alpha = {
            'solution': self.classical_population[best_idx],
            'fitness': self.fitness[best_idx]
        }
        self.convergence = [self.alpha['fitness']]
        
        for iteration in range(self.config['max_iterations']):
            # Update quantum states
            new_quantum_pop = []
            for q_state in self.quantum_population:
                # Quantum rotation towards alpha
                rotated = self._quantum_rotate(q_state, self.quantum_population[best_idx])
                
                # Quantum entanglement
                entangled = self._quantum_entangle(rotated)
                
                # Decaying rotation step
                self.config['rotation_step'] *= self.config['decay_rate']
                
                new_quantum_pop.append(entangled)
            
            self.quantum_population = new_quantum_pop
            
            # Measure new classical population
            self.classical_population = [
                self._quantum_measure(q_state)
                for q_state in self.quantum_population
            ]
            self.fitness = [self._calculate_distance(r) for r in self.classical_population]
            
            # Update alpha wolf
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.alpha['fitness']:
                self.alpha = {
                    'solution': self.classical_population[current_best_idx],
                    'fitness': self.fitness[current_best_idx]
                }
                best_idx = current_best_idx
            
            self.convergence.append(self.alpha['fitness'])
            
            # Diversity maintenance
            if len(set(map(tuple, self.classical_population))) < self.config['population_size']//2:
                num_new = self.config['population_size']//4
                new_states = [
                    np.random.uniform(0, 2*np.pi, 
                    (self.num_cities, self.config['quantum_bits']))
                    for _ in range(num_new)
                ]
                self.quantum_population[-num_new:] = new_states
            
            self.logger.info(
                f"Iter {iteration+1}/{self.config['max_iterations']}: " +
                f"Best={self.alpha['fitness']:.2f} " +
                f"Qubits={self.config['quantum_bits']}"
            )
        
        self._generate_outputs()
        return self.alpha

    def _generate_outputs(self):
        """Generate quantum optimization reports."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence, 'm-', linewidth=2)
        plt.title('Quantum-GWO Convergence', fontsize=14)
        plt.xlabel('Quantum Iterations', fontsize=12)
        plt.ylabel('Best Distance', fontsize=12)
        plt.grid(True, alpha=0.3)
        plot_path = f"{self.config['output_dir']}/convergence_{self.execution_id}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        report = f"""
        QUANTUM GWO OPTIMIZATION REPORT
        ===============================
        - Cities: {self.num_cities}
        - Qubits per City: {self.config['quantum_bits']}
        - Best Distance: {self.alpha['fitness']:.2f}
        - Optimal Route: {' → '.join(map(str, self.alpha['solution']))}
        - Final Rotation Step: {self.config['rotation_step']:.4f}
        - Entanglement Operations: {self.config['entanglement_prob']*100:.1f}%
        """
        print(report)
        with open(f"{self.config['output_dir']}/report_{self.execution_id}.txt", 'w') as f:
            f.write(report)

if __name__ == "__main__":
    # 5-city example with symmetrical distances
    dm = np.array([
        [0, 12, 10, 19, 8],
        [12, 0, 3, 7, 2],
        [10, 3, 0, 6, 9],
        [19, 7, 6, 0, 4],
        [8, 2, 9, 4, 0]
    ])
    
    qgwo = QuantumGWO_TSP(dm, {
        'population_size': 50,
        'max_iterations': 300,
        'quantum_bits': 10,
        'output_dir': 'quantum_gwo_demo'
    })
    result = qgwo.optimize()
    
    print(f"\nQuantum Optimal Route: {result['solution']}")
    print(f"Quantum Distance: {result['fitness']:.2f}")