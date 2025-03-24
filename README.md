# Quantum vs Classical Metaheuristics for Traveling Salesman Problem (TSP)

A rigorous comparative study of quantum-inspired and classical metaheuristic algorithms for solving the Traveling Salesman Problem (TSP), featuring parallel benchmarking, detailed analytics, and visualization capabilities.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Algorithms Implemented](#algorithms-implemented)
- [Installation Guide](#installation-guide)
- [Usage Examples](#usage-examples)
- [Configuration Options](#configuration-options)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Acknowledgments](#acknowledgments)

## Project Overview

This repository contains a production-grade implementation and benchmarking system for comparing classical and quantum-inspired metaheuristics on the NP-hard Traveling Salesman Problem. The study focuses on:

- **Algorithm Effectiveness**: Solution quality comparison
- **Convergence Behavior**: Optimization dynamics analysis
- **Computational Efficiency**: Runtime performance metrics
- **Quantum Advantages**: Exploration of quantum-inspired enhancements

The implementation supports reproducible research with seed management and deterministic problem generation.

## Key Features

### Algorithmic Implementations
- **Hybrid Architecture**: Base classes for easy extension
- **Quantum Operations**:
  - Quantum rotation gates
  - Entanglement operators
  - Quantum measurement simulation
- **Classical Metaheuristics**:
  - Population-based optimization
  - Adaptive parameter control
  - Diversity preservation mechanisms

### Benchmarking System
- **Parallel Execution**: Multi-threaded experiment runs
- **Statistical Analysis**:
  - Mean solution quality
  - Standard deviation
  - Convergence rates
- **Visualization**:
  - Convergence curves
  - Performance radar charts
  - Solution space mapping

### Technical Features
- **Configuration Management**: YAML/JSON support
- **Result Persistence**:
  - JSON reports
  - High-resolution plots
  - Optimization trajectories
- **Logging System**:
  - Multi-level logging (DEBUG, INFO, WARNING, ERROR)
  - File and console handlers

## Algorithms Implemented

| Algorithm | Type | Key Characteristics |
|-----------|------|---------------------|
| Classical BHO | Population-based | Event horizon replacement, stochastic mutation |
| Quantum-Inspired BHO | Quantum-enhanced | Quantum state rotation, probability amplitude measurement |
| Classical GWO | Swarm intelligence | Hierarchy-based search, crossover-dominated exploration |
| Quantum-Inspired GWO | Hybrid quantum-classical | Quantum-enhanced crossover, entanglement operators |

## Installation Guide

### Prerequisites
- Python 3.8+
- pip package manager

### Step-by-Step Setup

1. **Clone Repository**:
```bash
git clone https://github.com/SarangVehale/QuantumVsClassicalMetaheuristics_TSP_ComparativeStudy.git
cd QuantumVsClassicalMetaheuristics_TSP_ComparativeStudy
```

2. **Create Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate  # Windows
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify Installation**:
```bash
python -c "import numpy, matplotlib; print('Installation successful!')"
```

## Usage Examples

### Basic Benchmarking
```python
from tsp_benchmark import TSPBenchmark

bench_config = {
    'problem_size': 30,
    'population_size': 100,
    'max_iterations': 200,
    'runs': 10,
    'output_dir': 'my_benchmark',
    'algorithms': ['EnhancedTSPGWO', 'QuantumGWO_TSP']
}

benchmark = TSPBenchmark(bench_config)
benchmark.run(parallel=True)
```

### Individual Algorithm Execution
```python
from algorithms import QuantumGWO_TSP
import numpy as np

# Generate random distance matrix
num_cities = 20
dm = np.random.rand(num_cities, num_cities)
np.fill_diagonal(dm, 0)

# Configure and run optimizer
qgwo = QuantumGWO_TSP(dm, {
    'population_size': 50,
    'max_iterations': 100,
    'quantum_bits': 4,
    'output_dir': 'qgwo_results'
})

solution, distance = qgwo.solve()
print(f"Best tour: {solution}\nDistance: {distance:.2f}")
```

## Configuration Options

### Benchmark Configuration
```yaml
problem_size: 50            # Number of cities
population_size: 100        # Individuals per population
max_iterations: 500         # Iterations per run
runs: 20                    # Trials per algorithm
output_dir: "results"       # Output directory
seed: 42                    # Random seed
algorithms:                 # Algorithms to compare
  - EnhancedTSPGWO
  - QuantumGWO_TSP
  - TSPBHO
  - QIBHOTSP
```

### Algorithm Configuration (Example: Quantum GWO)
```yaml
quantum_bits: 8             # Qubits per city
entanglement_rate: 0.25     # Probability of entanglement
rotation_step: 0.05         # Rotation gate magnitude
mutation_rate: 0.15         # Classical mutation probability
a_decay: 2.0                # Exploration parameter decay
```

## Performance Metrics

| Metric | Description | Measurement Method |
|--------|-------------|---------------------|
| Solution Quality | Best found tour length | Mean ± SD across runs |
| Convergence Speed | Iterations to 95% of final solution | Curve derivative analysis |
| Computational Efficiency | CPU time per iteration | Timeit module |
| Population Diversity | Genotypic variance | Shannon entropy calculation |


## Contributing

We welcome contributions following these guidelines:

1. **Branch Naming**: `feature/[feature-name]` or `fix/[issue-name]`
2. **Code Standards**:
   - PEP8 compliance
   - Type hints for all functions
   - Docstrings following Google format
3. **Testing**:
   - 90%+ test coverage
   - pytest framework
4. **Documentation**:
   - Update README for new features
   - Add example notebooks

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf Optimizer. *Advances in Engineering Software*, 69, 46-61.
2. Hatamlou, A. (2013). Black hole: A new heuristic optimization approach for data clustering. *Information Sciences*, 222, 175-184.
3. Layeb, A. (2011). A novel quantum inspired cuckoo search algorithm for bin packing problem. *International Journal of Information Technology and Computer Science*, 4(5), 58-67.
4. Talbi, H., Draa, A., & Batouche, M. (2021). A survey on quantum-inspired computing: Concepts, algorithms, and recent developments. *Journal of King Saud University-Computer and Information Sciences*.

## Acknowledgments

- Inspired by the work of Dr. Seyedali Mirjalili on nature-inspired algorithms
- Quantum concepts adapted from Qiskit textbook
- Benchmarking methodology based on IEEE CEC guidelines

---

**Note**: This project is actively maintained. Report issues [here](https://github.com/SarangVehale/QuantumVsClassicalMetaheuristics_TSP_ComparativeStudy/issues).
```
