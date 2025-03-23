"""
QUANTUM-INSPIRED BLACK HOLE OPTIMIZER FOR TSP (FULLY CORRECTED)
================================================================
Robust implementation with proper file handling and error prevention.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional

class QuantumTSPOptimizer:
    """Quantum-inspired optimizer for TSP with robust file handling."""
    
    DEFAULT_CONFIG = {
        'quantum_bits': 10,
        'population_size': 50,
        'max_iterations': 200,
        'rotation_strength': 0.05,
        'entanglement_rate': 0.2,
        'output_dir': 'quantum_tsp_results',
        'log_level': logging.INFO
    }

    def __init__(self, distance_matrix: np.ndarray, config: Optional[dict] = None):
        """Initialize optimizer with proper setup sequence."""
        self._validate_inputs(distance_matrix)
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # Critical fix: Create directory before any file operations
        self._prepare_output_directory()
        self._setup_logging()
        
        self._initialize_quantum_states()
        self.best_solution = {'route': [], 'distance': float('inf')}
        self.history = []
        self.execution_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    def _validate_inputs(self, matrix: np.ndarray):
        """Validate TSP distance matrix integrity."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Distance matrix must be square")
        if matrix.shape[0] < 2:
            raise ValueError("At least 2 cities required")
        if np.any(matrix < 0):
            raise ValueError("Negative distances not allowed")
        np.fill_diagonal(matrix, 0)

    def _prepare_output_directory(self):
        """Ensure output directory exists before file operations."""
        os.makedirs(self.config['output_dir'], exist_ok=True)

    def _setup_logging(self):
        """Configure logging after directory creation."""
        logging.basicConfig(
            level=self.config['log_level'],
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config['output_dir']}/execution.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('QuantumTSP')
        self.logger.info(f"Logging initialized in {self.config['output_dir']}")

    def _initialize_quantum_states(self):
        """Initialize quantum population with random states."""
        self.quantum_states = [
            np.random.uniform(0, 2*np.pi, (self.num_cities, self.config['quantum_bits']))
            for _ in range(self.config['population_size'])
        ]
        self.logger.info(f"Initialized {self.config['population_size']} quantum solutions")

    def _quantum_measure(self, state: np.ndarray) -> List[int]:
        """Convert quantum state to classical route."""
        probabilities = (np.sin(state) ** 2).sum(axis=1)
        return np.argsort(probabilities).tolist()

    def _calculate_distance(self, route: List[int]) -> float:
        """Compute route distance with explicit loop for reliability."""
        total = 0.0
        for i in range(len(route)):
            current = route[i]
            next_city = route[(i + 1) % len(route)]
            total += self.distance_matrix[current, next_city]
        return total

    def _update_quantum_states(self, best_state: np.ndarray):
        """Update quantum states with guided operations."""
        new_states = []
        for state in self.quantum_states:
            # Rotate towards best solution
            rotated = state + self.config['rotation_strength'] * (best_state - state)
            # Apply entanglement
            if np.random.rand() < self.config['entanglement_rate']:
                i, j = np.random.choice(self.num_cities, 2, replace=False)
                rotated[[i, j]] = rotated[[j, i]]
            new_states.append(rotated)
        self.quantum_states = new_states

    def optimize(self) -> Dict:
        """Main optimization process with progress tracking."""
        self.logger.info("Starting quantum optimization process...")
        
        for iteration in range(self.config['max_iterations']):
            # Convert quantum states to classical routes
            classical_routes = [self._quantum_measure(s) for s in self.quantum_states]
            distances = [self._calculate_distance(r) for r in classical_routes]
            
            # Update best solution
            current_best_idx = np.argmin(distances)
            current_best = {
                'route': classical_routes[current_best_idx],
                'distance': distances[current_best_idx]
            }
            
            if current_best['distance'] < self.best_solution['distance']:
                self.best_solution = current_best
                self.logger.debug(f"Iter {iteration+1}: New best {current_best['distance']:.2f}")
            
            self.history.append(self.best_solution['distance'])
            self._update_quantum_states(self.quantum_states[current_best_idx])
        
        self._generate_outputs()
        return self.best_solution

    def _generate_outputs(self):
        """Generate comprehensive output reports."""
        # Convergence plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.history, 'b-', linewidth=2)
        plt.title('Optimization Progress', fontsize=14)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Best Distance', fontsize=12)
        plt.grid(True, alpha=0.3)
        plot_path = f"{self.config['output_dir']}/convergence_{self.execution_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Text report
        report = f"""
        QUANTUM TSP OPTIMIZATION REPORT
        ===============================
        - Number of Cities: {self.num_cities}
        - Best Route Distance: {self.best_solution['distance']:.2f}
        - Optimal Route: {' → '.join(map(str, self.best_solution['route']))}
        - Initial Distance: {self.history[0]:.2f}
        - Final Improvement: {((self.history[0] - self.best_solution['distance']) / self.history[0] * 100):.1f}%
        - Convergence Plot: {plot_path}
        """
        print(report)
        with open(f"{self.config['output_dir']}/report_{self.execution_id}.txt", 'w') as f:
            f.write(report)

if __name__ == "__main__":
    # Example 4-city problem
    distance_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    
    # Configure and run optimizer
    optimizer = QuantumTSPOptimizer(distance_matrix, {
        'population_size': 30,
        'max_iterations': 100,
        'output_dir': 'demo_results'
    })
    solution = optimizer.optimize()
    
    print(f"\nOptimal Route: {solution['route']}")
    print(f"Total Distance: {solution['distance']:.2f}")