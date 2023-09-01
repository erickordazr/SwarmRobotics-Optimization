# SwarmRobotics-Optimization

## Description

**SwarmRobotics-Optimization** is an advanced Python-based platform designed to simulate and optimize robot swarm behaviors. Building upon a foundational simulation, this enhanced version integrates the `pymoo` module to optimize swarm behaviors using NSGA III and MOEAD algorithms. Users can set parameters and visualize results, which are saved as numpy arrays for further analysis.

## Features

- **Enhanced Simulation**: Incorporates advanced optimization techniques to refine robot swarm behaviors.
- **NSGA III & MOEAD Integration**: Utilizes the `pymoo` module to implement NSGA III and MOEAD algorithms for optimization.
- **Visualization**: Option to visualize the simulation in real-time (if integrated with a visualization tool in the future).
- **Data Saving**: Results are saved as numpy arrays for post-processing and analysis.

## Setup and Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/erickordazr/SwarmRobotics-Optimization.git
    cd SwarmRobotics-Optimization
    ```

2. **Install Dependencies**:
    Ensure you have Python installed. Then, install the required packages using:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Enhanced Simulation**:
    Navigate to the `src` directory and run:
    ```bash
    python NSGA3.py
    python MOEAD.py
    ```

## Usage

1. **Input Parameters**:
    - After running `python NSGA3.py` or `python MOEAD.py`, you'll be prompted to input various parameters such as the number of objects, individuals, and radii for repulsion, orientation, and attraction.

2. **Choose Optimization Algorithm**:
    - Select between NSGA III and MOEAD for optimization.

3. **View Results**:
    - If you've chosen to visualize the simulation, watch it in real-time.
    - Once the simulation is complete, the results will be saved in the `data` directory as numpy arrays.

## Contributing

We welcome contributions! If you find a bug or have suggestions for improvements, please open an issue. If you'd like to contribute code, fork the repository, make your changes, and submit a pull request.

## License

This project is open-source and available under the [MIT License](LICENSE).
