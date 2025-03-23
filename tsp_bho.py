import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from typing import Optional, List, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TSPBHO')

DEFAULT_CONFIG = {
    'population_size': 50,
    'max_iterations': 100,
    'log_level': logging.INFO,
    'random_seed': None,
    'output_dir': 'results',
    'problem_name': 'TSP'
}

class TSPBHO:
    """Black Hole Optimizer for Traveling Salesman Problem with enhanced exploration."""
    
    def __init__(self, distance_matrix: np.ndarray, config: Optional[dict] = None):
        self._validate_distance_matrix(distance_matrix)
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        logger.setLevel(self.config['log_level'])
        self._prepare_output_dir()
        self.population = []
        self.distances = []
        self.best_tour = None
        self.best_distance = float('inf')
        self.convergence_curve = []
        self.rng = np.random.default_rng(self.config.get('random_seed'))
        self.execution_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def _prepare_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.config['output_dir'], exist_ok=True)

    def _validate_distance_matrix(self, distance_matrix):
        """Ensure the distance matrix is valid."""
        if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("Distance matrix must be square.")
        if np.any(distance_matrix < 0):
            raise ValueError("Distances must be non-negative.")
        if distance_matrix.shape[0] < 2:
            raise ValueError("At least two cities required.")
        np.fill_diagonal(distance_matrix, 0)
        
    def _initialize_population(self):
        """Generate initial random population of tours."""
        self.population = [self.rng.permutation(self.num_cities).tolist()
                          for _ in range(self.config['population_size'])]
        
    def _evaluate_tour(self, tour: List[int]) -> float:
        """Calculate the total tour distance using vectorized operations."""
        tour_arr = np.array(tour)
        shifted = np.roll(tour_arr, -1)
        return np.sum(self.distance_matrix[tour_arr, shifted])
    
    def _move_star(self, star: List[int]) -> List[int]:
        """Move a star towards the black hole using Ordered Crossover (OX)."""
        size = len(star)
        start, end = sorted(self.rng.choice(size, size=2, replace=False))
        child = [-1] * size
        
        # Copy segment from the black hole
        child[start:end] = self.best_tour[start:end]
        
        # Fill remaining positions from the star's tour
        current_pos = 0
        for i in range(size):
            if not (start <= i < end):
                while True:
                    city = star[current_pos % size]
                    if city not in child:
                        child[i] = city
                        current_pos += 1
                        break
                    current_pos += 1
        return child
    
    def _absorb_stars(self, R: float) -> Tuple[List[List[int]], List[float]]:
        """Improved absorption criteria with normalized probabilities."""
        normalized_dists = (np.array(self.distances) - self.best_distance) / (max(self.distances) - self.best_distance + 1e-10)
        absorption_probs = normalized_dists * R
        
        new_population = []
        new_distances = []
        for i, (tour, dist) in enumerate(zip(self.population, self.distances)):
            if tour == self.best_tour:
                new_population.append(tour)
                new_distances.append(dist)
                continue
            
            if self.rng.random() < absorption_probs[i]:
                new_tour = self.rng.permutation(self.num_cities).tolist()
                new_dist = self._evaluate_tour(new_tour)
                new_population.append(new_tour)
                new_distances.append(new_dist)
            else:
                new_population.append(tour)
                new_distances.append(dist)
        return new_population, new_distances

    def solve(self) -> Tuple[List[int], float]:
        """Run the BHO optimization to solve the TSP."""
        self._initialize_population()
        self.distances = [self._evaluate_tour(t) for t in self.population]
        best_idx = np.argmin(self.distances)
        self.best_tour = self.population[best_idx]
        self.best_distance = self.distances[best_idx]
        self.convergence_curve = [self.best_distance]
        logger.info(f"Initial best distance: {self.best_distance}")
        
        for iteration in range(self.config['max_iterations']):
            # Move each star towards the black hole
            new_population = []
            new_distances = []
            for tour in self.population:
                new_tour = self._move_star(tour)
                new_dist = self._evaluate_tour(new_tour)
                new_population.append(new_tour)
                new_distances.append(new_dist)
            self.population = new_population
            self.distances = new_distances
            
            # Update the best tour after movement
            current_best_idx = np.argmin(self.distances)
            current_best_dist = self.distances[current_best_idx]
            if current_best_dist < self.best_distance:
                self.best_distance = current_best_dist
                self.best_tour = self.population[current_best_idx]
            
            # Dynamic absorption probability
            avg_distance = np.mean(self.distances)
            R = 0.2 + 0.6 * (iteration / self.config['max_iterations'])
            
            # Improved absorption with diversity control
            self.population, self.distances = self._absorb_stars(R)
            
            # Update the best tour after absorption
            current_best_idx = np.argmin(self.distances)
            current_best_dist = self.distances[current_best_idx]
            if current_best_dist < self.best_distance:
                self.best_distance = current_best_dist
                self.best_tour = self.population[current_best_idx]
            
            self.convergence_curve.append(self.best_distance)
            logger.info(f"Iteration {iteration+1}/{self.config['max_iterations']}: Best Distance: {self.best_distance}")
        
        return self.best_tour, self.best_distance
    
    def _save_plot(self, fig, plot_name: str):
        """Save plot to file with timestamp."""
        filename = f"{self.config['problem_name']}_{plot_name}_{self.execution_timestamp}.png"
        path = os.path.join(self.config['output_dir'], filename)
        fig.savefig(path, dpi=300)
        logger.info(f"Saved plot: {path}")

    def plot_convergence(self):
        """Plot and save convergence curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve, marker='o', linestyle='-', color='b')
        plt.title('Algorithm Convergence Progress')
        plt.xlabel('Iteration Number')
        plt.ylabel('Shortest Found Route Distance')
        plt.grid(True)
        self._save_plot(plt.gcf(), 'convergence')
        plt.close()

    def plot_route(self, city_coords: Optional[np.ndarray] = None):
        """Visualize and save TSP route."""
        plt.figure(figsize=(10, 6))
        
        if city_coords is None:
            city_coords = self.rng.random((self.num_cities, 2)) * 100
            plt.title('Best TSP Route (Random City Layout)')
        else:
            plt.title('Best TSP Route')
        
        plt.scatter(city_coords[:, 0], city_coords[:, 1], c='red', s=100, label='Cities')
        
        tour = self.best_tour + [self.best_tour[0]]
        for i in range(len(tour) - 1):
            start, end = tour[i], tour[i+1]
            plt.plot([city_coords[start, 0], city_coords[end, 0]],
                     [city_coords[start, 1], city_coords[end, 1]], 'b-')
        
        plt.xlabel('X Coordinate (km)')
        plt.ylabel('Y Coordinate (km)')
        plt.legend()
        plt.grid(True)
        self._save_plot(plt.gcf(), 'route')
        plt.close()

    def generate_report(self, city_coords: Optional[np.ndarray] = None):
        """Generate comprehensive results summary."""
        print("\n=== TSP SOLUTION REPORT ===")
        print(f"Optimal Route Distance: {self.best_distance:.2f} units")
        print(f"Number of Cities: {self.num_cities}")
        print(f"Computation Time: {self.execution_timestamp}")
        print("\nRoute Sequence:")
        print(" -> ".join(str(city) for city in self.best_tour) + f" -> {self.best_tour[0]}")
        
        if city_coords is not None:
            total_distance = self._calculate_actual_distance(city_coords)
            print(f"\nGeographic Validation:")
            print(f"Calculated Distance from Coordinates: {total_distance:.2f} units")
        
        print("\nAlgorithm Performance:")
        improvement = ((self.convergence_curve[0] - self.best_distance) / self.convergence_curve[0]) * 100
        print(f"Initial Best Distance: {self.convergence_curve[0]:.2f}")
        print(f"Final Best Distance: {self.best_distance:.2f}")
        print(f"Improvement: {improvement:.1f}%")
        print(f"Iterations Completed: {len(self.convergence_curve)}")
        print("\nOutput Files:")
        print(f"Convergence plot saved to: {self.config['output_dir']}/convergence*.png")
        print(f"Route visualization saved to: {self.config['output_dir']}/route*.png")

    def _calculate_actual_distance(self, city_coords: np.ndarray) -> float:
        """Calculate actual geographic distance for validation."""
        total = 0.0
        for i in range(len(self.best_tour)):
            current = self.best_tour[i]
            next_city = self.best_tour[(i+1)%len(self.best_tour)]
            dx = city_coords[next_city, 0] - city_coords[current, 0]
            dy = city_coords[next_city, 1] - city_coords[current, 1]
            total += np.sqrt(dx**2 + dy**2)
        return total

# Example usage
if __name__ == "__main__":
    # Create sample city coordinates
    cities = {
        'A': (0, 0),
        'B': (10, 0),
        'C': (15, 10),
        'D': (5, 15)
    }
    
    # Create distance matrix from coordinates
    city_coords = np.array(list(cities.values()))
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))
    
    for i in range(num_cities):
        for j in range(num_cities):
            dx = city_coords[j,0] - city_coords[i,0]
            dy = city_coords[j,1] - city_coords[i,1]
            distance_matrix[i,j] = np.sqrt(dx**2 + dy**2)
    
    # Configure solver
    config = {
        'population_size': 100,
        'max_iterations': 200,
        'problem_name': '4City_TSP',
        'output_dir': 'tsp_results'
    }
    
    # Run optimization
    solver = TSPBHO(distance_matrix, config)
    best_tour, best_distance = solver.solve()
    
    # Generate outputs
    solver.generate_report(city_coords)
    solver.plot_convergence()
    solver.plot_route(city_coords)