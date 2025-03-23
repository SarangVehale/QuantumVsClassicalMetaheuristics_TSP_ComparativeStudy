import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up a system to show messages while the program runs.
# These messages help you see what’s happening step by step, like a progress report.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BlackHoleOptimizer:
    """
    This is a tool to find the shortest route for visiting a list of cities and returning home.
    It’s inspired by black holes in space! Here’s how it works:
    - It creates many possible routes (called "stars").
    - It picks the best route as the "black hole."
    - It moves other routes closer to the black hole to make them better.
    - If a route gets too close, it’s replaced with a new one to keep things fresh.
    """

    def __init__(self, distance_matrix=None, coordinates=None, population_size=50, max_iterations=1000, random_seed=None):
        """
        Set up the tool with either a distance table or city locations.

        What you need to provide:
            - distance_matrix: A table showing how far each city is from every other city (optional).
            - coordinates: A list of city locations, like (x, y) points on a map (optional).
            - population_size: How many routes to try at once (default is 50).
            - max_iterations: How many times to improve the routes (default is 1000).
            - random_seed: A number to make the results the same each time (optional).

        If something’s wrong with the inputs, it will tell you!
        """
        # Check if we have a distance table or city locations
        if distance_matrix is not None:
            # Make sure the distance table is a square (same number of rows and columns)
            if not isinstance(distance_matrix, np.ndarray) or distance_matrix.shape[0] != distance_matrix.shape[1]:
                raise ValueError("Distance table must be a square (like 3x3 for 3 cities).")
            self.distance_matrix = distance_matrix
            self.n = distance_matrix.shape[0]  # Number of cities
            self.coordinates = None
        elif coordinates is not None:
            # Make sure coordinates are a list with at least 2 cities
            if not isinstance(coordinates, list) or len(coordinates) < 2:
                raise ValueError("Coordinates must be a list of at least two (x, y) points.")
            self.coordinates = np.array(coordinates)
            self.n = len(coordinates)
            self.distance_matrix = self.compute_distance_matrix()  # Calculate distances from locations
        else:
            raise ValueError("Please provide either a distance table or city locations.")

        # Check that population_size and max_iterations are sensible numbers
        if not isinstance(population_size, int) or population_size < 1:
            raise ValueError("Population size must be a positive number (like 50).")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("Max iterations must be a positive number (like 1000).")

        # Save the settings
        self.population_size = population_size
        self.max_iterations = max_iterations
        if random_seed is not None:
            np.random.seed(random_seed)  # Make results repeatable

    def compute_distance_matrix(self):
        """
        Figure out the distance between every pair of cities using their locations.
        Returns a table of distances.
        """
        dist = np.zeros((self.n, self.n))  # Start with a blank table
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Calculate distance using the formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)
                d = np.linalg.norm(self.coordinates[i] - self.coordinates[j])
                dist[i, j] = d
                dist[j, i] = d  # Distance is the same both ways
        return dist

    def compute_tour_length(self, permutation):
        """
        Add up the total distance for a route that visits cities in a specific order.

        Example: If permutation is [0, 1, 2], it calculates distance from 0 to 1, 1 to 2, and 2 back to 0.
        Returns the total distance.
        """
        length = 0
        for i in range(len(permutation) - 1):
            length += self.distance_matrix[permutation[i], permutation[i + 1]]
        length += self.distance_matrix[permutation[-1], permutation[0]]  # Back to the start
        return length

    def decode_solution(self, x):
        """
        Turn a list of random numbers into an order of cities to visit.
        Example: If x = [0.3, 0.1, 0.4], it becomes [1, 0, 2] (sorted order of positions).
        Returns the city order.
        """
        return np.argsort(x)  # Sort the numbers and give their positions

    def compute_fitness(self, tour_length):
        """
        Decide how good a route is. Shorter routes are better, so fitness = 1 / distance.
        A smaller distance means a higher fitness score.
        Returns the fitness value.
        """
        return 1 / tour_length

    def run(self):
        """
        Run the tool to find the best route!
        Returns:
            - best_solution: The order of cities for the shortest route.
            - best_tour_length: The total distance of that route.
            - convergence: A list showing how the best distance improved over time.
        """
        logging.info("Starting to find the shortest route!")
        
        # Step 1: Create a bunch of random routes (stars)
        x = np.random.rand(self.population_size, self.n)  # Random numbers for each route
        permutations = [self.decode_solution(x[i]) for i in range(self.population_size)]
        tour_lengths = [self.compute_tour_length(p) for p in permutations]
        fitnesses = [self.compute_fitness(tl) for tl in tour_lengths]
        
        # Step 2: Find the best route (black hole) to start with
        bh_index = np.argmax(fitnesses)  # The route with the highest fitness
        best_tour_length = tour_lengths[bh_index]
        convergence = [best_tour_length]  # Keep track of the best distance

        # Step 3: Improve routes over many steps
        for iteration in range(self.max_iterations):
            # Move all routes (except the black hole) closer to the black hole
            update_mask = np.arange(self.population_size) != bh_index
            rand = np.random.rand(sum(update_mask))  # Random amounts to move
            x[update_mask] = x[update_mask] + rand[:, None] * (x[bh_index] - x[update_mask])

            # Update the routes, distances, and fitnesses
            for i in np.where(update_mask)[0]:
                permutations[i] = self.decode_solution(x[i])
                tour_lengths[i] = self.compute_tour_length(permutations[i])
                fitnesses[i] = self.compute_fitness(tour_lengths[i])

            # Check if a new route beats the black hole
            new_bh_index = np.argmax(fitnesses)
            if fitnesses[new_bh_index] > fitnesses[bh_index]:
                bh_index = new_bh_index
                best_tour_length = tour_lengths[bh_index]

            # Set a boundary (event horizon) around the black hole
            sum_fitness = sum(fitnesses)
            R = fitnesses[bh_index] / sum_fitness if sum_fitness > 0 else 0

            # Replace routes that get too close to the black hole
            for i in range(self.population_size):
                if i != bh_index:
                    distance = np.linalg.norm(x[i] - x[bh_index])
                    if distance < R:
                        x[i] = np.random.rand(self.n)  # New random route
                        permutations[i] = self.decode_solution(x[i])
                        tour_lengths[i] = self.compute_tour_length(permutations[i])
                        fitnesses[i] = self.compute_fitness(tour_lengths[i])

            convergence.append(best_tour_length)
            logging.info(f"Step {iteration + 1}/{self.max_iterations}: Best distance so far = {best_tour_length:.4f}")

        best_solution = permutations[bh_index]
        logging.info("Finished! Found the shortest route.")
        return best_solution, best_tour_length, convergence

    def plot_convergence(self, convergence):
        """
        Draw a line graph showing how the best route distance got better over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(convergence, label='Best Route Distance')
        plt.xlabel('Step Number')
        plt.ylabel('Best Route Distance')
        plt.title('How the Route Distance Improved')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_route(self, permutation):
        """
        Draw a map of the best route if we have city locations.
        Shows the path connecting the cities in order.
        """
        if self.coordinates is None:
            logging.warning("Can’t draw a map without city locations.")
            return
        
        plt.figure(figsize=(10, 6))
        ordered_coords = self.coordinates[permutation]
        # Draw the route
        plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'o-', label='Route')
        # Connect the last city back to the first
        plt.plot([ordered_coords[-1, 0], ordered_coords[0, 0]], 
                 [ordered_coords[-1, 1], ordered_coords[0, 1]], 'o-')
        # Label each city with its number
        for i, (x, y) in enumerate(ordered_coords):
            plt.text(x, y, str(permutation[i]), fontsize=12, ha='right')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Best Route on a Map')
        plt.legend()
        plt.grid(True)
        plt.show()

# Try it out!
if __name__ == "__main__":
    # Example 1: Using city locations (a square)
    coordinates = [(0, 0), (1, 0), (1, 1), (0, 1)]
    bho = BlackHoleOptimizer(coordinates=coordinates, population_size=20, max_iterations=100, random_seed=42)
    best_solution, best_tour_length, convergence = bho.run()
    print(f"Best Order to Visit Cities: {best_solution}")
    print(f"Total Distance of the Best Route: {best_tour_length:.4f}")
    bho.plot_convergence(convergence)
    bho.plot_route(best_solution)

    # Example 2: Using a distance table
    distance_matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    bho = BlackHoleOptimizer(distance_matrix=distance_matrix, population_size=10, max_iterations=50, random_seed=42)
    best_solution, best_tour_length, convergence = bho.run()
    print(f"Best Order to Visit Cities: {best_solution}")
    print(f"Total Distance of the Best Route: {best_tour_length:.4f}")
    bho.plot_convergence(convergence)
    bho.plot_route(best_solution)  # This will warn you since there are no locations