"""
PRODUCTION-GRADE TSP OPTIMIZATION SUITE
=======================================
Features:
- Classical & Quantum-inspired BHO/GWO algorithms
- Comprehensive configuration validation
- Detailed logging and metrics tracking
- Parallel benchmarking capabilities
- Result visualization and persistence
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Callable
from collections import defaultdict
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TSPOptimizer(ABC):
    """Abstract base class for TSP optimizers"""
    
    DEFAULT_CONFIG = {
        'max_iterations': 100,
        'population_size': 50,
        'output_dir': 'results',
        'log_level': logging.INFO,
        'seed': None
    }

    def __init__(self, distance_matrix: np.ndarray, config: Optional[dict] = None):
        self._validate_input(distance_matrix)
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.config = self._merge_config(config)
        self.convergence = []
        self.execution_time = 0.0
        self._prepare_environment()

    def _validate_input(self, matrix: np.ndarray):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Invalid distance matrix: must be square 2D array")
        if not np.issubdtype(matrix.dtype, np.number):
            raise ValueError("Distance matrix must contain numeric values")

    def _merge_config(self, config: Optional[dict]) -> dict:
        merged = self.DEFAULT_CONFIG.copy()
        if config:
            merged.update(config)
        return merged

    def _prepare_environment(self):
        os.makedirs(self.config['output_dir'], exist_ok=True)
        logging.getLogger().setLevel(self.config['log_level'])
        if self.config['seed'] is not None:
            np.random.seed(self.config['seed'])

    @abstractmethod
    def solve(self) -> Tuple[List[int], float]:
        pass

    def _calculate_distance(self, tour: List[int]) -> float:
        return sum(
            self.distance_matrix[tour[i], tour[(i+1) % self.num_cities]]
            for i in range(self.num_cities)
        )

    def save_results(self):
        """Save convergence plot and final solution"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence)
        plt.title(f'{self.__class__.__name__} Convergence')
        plt.xlabel('Iterations')
        plt.ylabel('Best Distance')
        plt.savefig(Path(self.config['output_dir']) / f'{self.__class__.__name__}_convergence.png')
        plt.close()

        with open(Path(self.config['output_dir']) / f'{self.__class__.__name__}_solution.json', 'w') as f:
            json.dump({
                'solution': self.best_solution,
                'distance': self.best_distance,
                'execution_time': self.execution_time,
                'convergence': self.convergence
            }, f)

# ==================================================================
# 1. CLASSICAL BLACK HOLE OPTIMIZER (BHO)
# ==================================================================
class TSPBHO(TSPOptimizer):
    """Black Hole Optimization implementation for TSP"""
    
    DEFAULT_CONFIG = {
        **TSPOptimizer.DEFAULT_CONFIG,
        'event_horizon': 0.25,
        'output_dir': 'bho_results'
    }

    def solve(self) -> Tuple[List[int], float]:
        start_time = time.time()
        population = self._initialize_population()
        self.best_solution, self.best_distance = self._find_best(population)
        self.convergence.append(self.best_distance)

        for _ in range(self.config['max_iterations']):
            population = self._move_stars(population)
            black_hole = self._create_black_hole(population)
            population = self._handle_event_horizon(population, black_hole)
            
            current_best, current_dist = self._find_best(population)
            if current_dist < self.best_distance:
                self.best_solution = current_best
                self.best_distance = current_dist
            self.convergence.append(self.best_distance)

        self.execution_time = time.time() - start_time
        self.save_results()
        return self.best_solution, self.best_distance

    def _initialize_population(self) -> List[List[int]]:
        return [np.random.permutation(self.num_cities).tolist()
                for _ in range(self.config['population_size'])]

    def _find_best(self, population: List[List[int]]) -> Tuple[List[int], float]:
        distances = [self._calculate_distance(tour) for tour in population]
        best_idx = np.argmin(distances)
        return population[best_idx], distances[best_idx]

    def _move_stars(self, population: List[List[int]]) -> List[List[int]]:
        return [self._mutate(tour) for tour in population]

    def _mutate(self, tour: List[int]) -> List[int]:
        i, j = np.random.choice(len(tour), 2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]
        return tour

    def _create_black_hole(self, population: List[List[int]]) -> List[int]:
        return self.best_solution.copy()

    def _handle_event_horizon(self, population: List[List[int]], 
                            black_hole: List[int]) -> List[List[int]]:
        new_pop = []
        for tour in population:
            if (self._calculate_distance(tour) / self.best_distance) < self.config['event_horizon']:
                new_pop.append(np.random.permutation(self.num_cities).tolist())
            else:
                new_pop.append(tour)
        return new_pop

# ==================================================================
# 2. QUANTUM-INSPIRED BHO (QBHO) 
# ==================================================================
class QIBHOTSP(TSPBHO):
    """Quantum-Inspired Black Hole Optimization with Dimension Fix"""
    
    DEFAULT_CONFIG = {
        **TSPBHO.DEFAULT_CONFIG,
        'quantum_bits': 8,
        'rotation_step': 0.1,
        'output_dir': 'qbho_results'
    }

    def __init__(self, distance_matrix: np.ndarray, config: Optional[dict] = None):
        super().__init__(distance_matrix, config)
        self.tour_to_index = {}

    def _initialize_population(self) -> List[List[int]]:
        self.quantum_states = [
            np.random.uniform(0, 2*np.pi, (self.num_cities, self.config['quantum_bits']))
            for _ in range(self.config['population_size'])
        ]
        population = []
        for idx, state in enumerate(self.quantum_states):
            tour = self._measure(state)
            self.tour_to_index[tuple(tour)] = idx
            population.append(tour)
        return population

    def _mutate(self, tour: List[int]) -> List[int]:
        # Get corresponding quantum state index
        idx = self.tour_to_index.get(tuple(tour), np.random.randint(len(self.quantum_states)))
        
        # Ensure we're using quantum states for calculations
        current_state = self.quantum_states[idx]
        target_state = self.quantum_states[np.random.randint(len(self.quantum_states))]
        
        # Perform quantum rotation
        rotated = current_state + self.config['rotation_step'] * (target_state - current_state)
        
        # Update state and mapping
        new_tour = self._measure(rotated)
        self.quantum_states[idx] = rotated
        self.tour_to_index[tuple(new_tour)] = idx
        
        return new_tour

    def _measure(self, state: np.ndarray) -> List[int]:
        probabilities = (np.sin(state) ** 2).sum(axis=1)
        return np.argsort(probabilities).tolist()

# ==================================================================
# 3. ENHANCED GREY WOLF OPTIMIZER (GWO)
# ==================================================================
class EnhancedTSPGWO(TSPOptimizer):
    """Enhanced Grey Wolf Optimizer with adaptive parameters"""
    
    DEFAULT_CONFIG = {
        **TSPOptimizer.DEFAULT_CONFIG,
        'a_decay': 2.0,
        'crossover_rate': 0.85,
        'mutation_rate': 0.15,
        'output_dir': 'gwo_results'
    }

    def solve(self) -> Tuple[List[int], float]:
        start_time = time.time()
        self.population = self._initialize_population()
        self._update_hierarchy()
        self.convergence = [self.alpha[1]]

        for iteration in range(self.config['max_iterations']):
            a = 2.0 - iteration * (2.0 / self.config['max_iterations'])
            
            new_population = []
            for wolf in self.population:
                if np.random.rand() < self.config['crossover_rate']:
                    child = self._crossover(wolf, self.alpha[0])
                    child = self._mutate(child)
                    new_population.append(child)
                else:
                    new_population.append(wolf)
            
            self.population = new_population
            self._update_hierarchy()
            self.convergence.append(self.alpha[1])

        self.best_solution, self.best_distance = self.alpha
        self.execution_time = time.time() - start_time
        self.save_results()
        return self.alpha

    def _initialize_population(self) -> List[List[int]]:
        return [np.random.permutation(self.num_cities).tolist()
                for _ in range(self.config['population_size'])]

    def _update_hierarchy(self):
        ranked = sorted([(tour, self._calculate_distance(tour)) 
                        for tour in self.population], key=lambda x: x[1])
        self.alpha = ranked[0]
        self.beta = ranked[1]
        self.delta = ranked[2]

    def _crossover(self, parent: List[int], alpha: List[int]) -> List[int]:
        size = len(parent)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        child = [-1]*size
        child[start:end] = parent[start:end]
        
        current = 0
        for i in range(size):
            if not start <= i < end:
                while True:
                    gene = alpha[current % size]
                    if gene not in child:
                        child[i] = gene
                        current += 1
                        break
                    current += 1
        return child

    def _mutate(self, tour: List[int]) -> List[int]:
        if np.random.rand() < self.config['mutation_rate']:
            i, j = np.random.choice(len(tour), 2, replace=False)
            tour[i], tour[j] = tour[j], tour[i]
        return tour

# ==================================================================
# 4. QUANTUM-INSPIRED GWO (QGWO) - CORRECTED
# ==================================================================
class QuantumGWO_TSP(EnhancedTSPGWO):
    """Quantum-Inspired Grey Wolf Optimizer"""
    
    DEFAULT_CONFIG = {
        **EnhancedTSPGWO.DEFAULT_CONFIG,
        'quantum_bits': 8,
        'entanglement_rate': 0.2,
        'output_dir': 'qgwo_results'
    }

    def _initialize_population(self) -> List[List[int]]:
        self.quantum_states = [
            np.random.uniform(0, 2*np.pi, (self.num_cities, self.config['quantum_bits']))
            for _ in range(self.config['population_size'])]
        return [self._measure(state) for state in self.quantum_states]

    def _measure(self, state: np.ndarray) -> List[int]:
        probabilities = (np.sin(state) ** 2).sum(axis=1)
        return np.argsort(probabilities).tolist()

    def _crossover(self, parent: List[int], alpha: List[int]) -> List[int]:
        base_child = super()._crossover(parent, alpha)
        if np.random.rand() < self.config['entanglement_rate']:
            i, j = np.random.choice(len(base_child), 2, replace=False)
            base_child[i], base_child[j] = base_child[j], base_child[i]
        return base_child

# ==================================================================
# 5. BENCHMARKING SYSTEM
# ==================================================================
class TSPBenchmark:
    """Comparative benchmarking system for TSP optimizers"""
    
    DEFAULT_CONFIG = {
        'problem_size': 50,
        'population_size': 100,
        'max_iterations': 500,
        'runs': 10,
        'output_dir': 'benchmark_results',
        'algorithms': [EnhancedTSPGWO, QuantumGWO_TSP, TSPBHO, QIBHOTSP]
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.distance_matrix = self._generate_problem()
        self.results = defaultdict(lambda: {
            'solutions': [],
            'distances': [],
            'times': [],
            'convergence': []
        })
        self._prepare_environment()

    def _prepare_environment(self):
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(Path(self.config['output_dir']) / 'benchmark.log'),
                logging.StreamHandler()
            ]
        )

    def _generate_problem(self) -> np.ndarray:
        np.random.seed(42)
        locations = np.random.rand(self.config['problem_size'], 2)
        return np.sqrt(((locations[:, None] - locations)**2).sum(axis=2))

    def run(self, parallel: bool = True):
        logger.info("Starting TSP benchmarking process")
        start_time = time.time()
        
        if parallel:
            with ThreadPoolExecutor() as executor:
                futures = []
                for algo in self.config['algorithms']:
                    futures += [executor.submit(self._run_single, algo, run)
                              for run in range(self.config['runs'])]
                
                for future in futures:
                    algo_name, result = future.result()
                    self._record_result(algo_name, result)
        else:
            for algo in self.config['algorithms']:
                for run in range(self.config['runs']):
                    algo_name, result = self._run_single(algo, run)
                    self._record_result(algo_name, result)

        self._generate_report()
        logger.info(f"Benchmarking completed in {time.time()-start_time:.2f}s")

    def _run_single(self, algorithm: Callable, run_id: int) -> Tuple[str, dict]:
        try:
            solver = algorithm(self.distance_matrix, {
                'population_size': self.config['population_size'],
                'max_iterations': self.config['max_iterations'],
                'output_dir': self.config['output_dir'],
                'seed': run_id
            })
            solution, distance = solver.solve()
            return algorithm.__name__, {
                'solution': solution,
                'distance': distance,
                'time': solver.execution_time,
                'convergence': solver.convergence
            }
        except Exception as e:
            logger.error(f"Error in {algorithm.__name__} run {run_id}: {str(e)}")
            return algorithm.__name__, None

    def _record_result(self, algo_name: str, result: Optional[dict]):
        if result is None:
            return
        self.results[algo_name]['solutions'].append(result['solution'])
        self.results[algo_name]['distances'].append(result['distance'])
        self.results[algo_name]['times'].append(result['time'])
        self.results[algo_name]['convergence'].append(result['convergence'])

    def _generate_report(self):
        report = {
            'metadata': {
                'problem_size': self.config['problem_size'],
                'runs': self.config['runs'],
                'timestamp': datetime.now().isoformat()
            },
            'results': {}
        }

        plt.figure(figsize=(12, 8))
        for algo, data in self.results.items():
            distances = np.array(data['distances'])
            times = np.array(data['times'])
            
            report['results'][algo] = {
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances),
                'mean_time': np.mean(times),
                'std_time': np.std(times)
            }

            avg_convergence = np.mean(data['convergence'], axis=0)
            plt.plot(avg_convergence, label=algo)

        plt.title('Algorithm Convergence Comparison')
        plt.xlabel('Iterations')
        plt.ylabel('Distance')
        plt.legend()
        plt.savefig(Path(self.config['output_dir']) / 'convergence_comparison.png')
        plt.close()

        with open(Path(self.config['output_dir']) / 'benchmark_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        logger.info("\nBenchmark Summary:")
        for algo, stats in report['results'].items():
            logger.info(
                f"{algo}: "
                f"Distance {stats['mean_distance']:.2f} ± {stats['std_distance']:.2f} | "
                f"Time {stats['mean_time']:.2f}s ± {stats['std_time']:.2f}"
            )

if __name__ == "__main__":
    benchmark = TSPBenchmark({
        'problem_size': 30,
        'population_size': 100,
        'max_iterations': 200,
        'runs': 5
    })
    benchmark.run(parallel=True)