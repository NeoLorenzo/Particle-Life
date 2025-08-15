# main.py
"""
Main entry point for the Particle Life simulation.

This script orchestrates the entire simulation lifecycle:
1. Loads configuration from `config.json`.
2. Initializes the logging system.
3. Sets up the simulation environment and particles.
4. Runs the main simulation loop.
5. Handles clean shutdown.
"""
import logging
from utils import setup_logging, load_config
import numpy as np

def main():
    """
    The main function to run the simulation.
    """
    # Load configuration from the JSON file first.
    # Logging is not set up yet, so we use a print for this one error.
    try:
        config = load_config('config.json')
    except Exception as e:
        print(f"FATAL: Could not load config.json. Error: {e}")
        return

    # Set up the logging system based on the loaded configuration.
    setup_logging(config)

    logging.info("--- Particle Life Simulation Starting ---")

    # Rule 7 (DIP): Depend on abstractions. We get config sections.
    sim_params = config.get('simulation_parameters', {})
    run_params = config.get('run_control', {})

    # Rule 1 (Application Constants): Import and use constants.
    from constants import WINDOW_WIDTH, WINDOW_HEIGHT
    from particle import ParticleSystem
    from simulation import Simulation

    # Initialize simulation components
    particles = ParticleSystem(sim_params, WINDOW_WIDTH, WINDOW_HEIGHT)
    sim = Simulation(particles, sim_params)

    # Initialize the visualizer
    from visualization import Visualizer
    visualizer = Visualizer(WINDOW_WIDTH, WINDOW_HEIGHT)

    # Main simulation loop
    log_throttle = run_params.get('log_throttle_steps', 100)
    
    running = True
    step_num = 0
    while running:
        sim.step()
        step_num += 1

        # The visualizer's draw method now controls the loop
        # by checking for the QUIT event.
        running = visualizer.draw(particles)

        # Rule 2.4: Hot loops must throttle logs
        if step_num % log_throttle == 0:
            logging.info(f"Simulation step {step_num}")
            
            # Example of an aggregated metric for DEBUG logging
            avg_velocity = np.mean(np.linalg.norm(particles.velocities, axis=1))
            logging.debug(f"Step {step_num} | Average Velocity: {avg_velocity:.4f}")

    visualizer.close()
    logging.info("Simulation loop finished.")
    logging.info("--- Particle Life Simulation Shutting Down ---")


if __name__ == "__main__":
    main()