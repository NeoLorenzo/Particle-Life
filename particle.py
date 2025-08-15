# particle.py
"""
Manages the state of all particles in the simulation.

This module defines the ParticleSystem class, which is responsible for
initializing and storing particle data (e.g., position, velocity, type)
in efficient NumPy arrays.
"""
import logging
import numpy as np
from typing import Dict, Any

# --- Data Contracts ---
#
# class ParticleSystem:
#   - __init__(self, params: Dict[str, Any], width: int, height: int):
#     - Inputs:
#       - params: Dictionary of simulation parameters from config.json.
#         - "seed": int
#         - "particle_count": int
#         - "particle_types": int
#       - width: int, width of the simulation world.
#       - height: int, height of the simulation world.
#     - Outputs: None
#     - Side Effects: Initializes internal NumPy arrays for particle state.
#     - Invariants:
#       - self.positions is a NumPy array of shape (N, 2) of dtype float64.
#       - self.velocities is a NumPy array of shape (N, 2) of dtype float64.
#       - self.types is a NumPy array of shape (N,) of dtype int32.

class ParticleSystem:
    """
    A container for all particles, managing their state via NumPy arrays.
    """
    def __init__(self, params: Dict[str, Any], width: int, height: int):
        """
        Initializes the particle system.

        Args:
            params (Dict[str, Any]): Simulation parameters from config.
            width (int): The width of the simulation area.
            height (int): The height of the simulation area.
        """
        self.particle_count = params['particle_count']
        self.particle_types = params['particle_types']
        self.seed = params['seed']

        # Rule 12: All randomness is controlled by a single master seed.
        # We create a dedicated RNG from the seed for all operations
        # within this module.
        self.rng = np.random.default_rng(self.seed)

        # Initialize particle state arrays
        self.positions = self.rng.uniform(
            low=[0, 0],
            high=[width, height],
            size=(self.particle_count, 2)
        )
        self.velocities = np.zeros((self.particle_count, 2), dtype=np.float64)
        self.types = self.rng.integers(
            low=0,
            high=self.particle_types,
            size=self.particle_count,
            dtype=np.int32
        )

        logging.info(
            f"ParticleSystem initialized with {self.particle_count} "
            f"particles of {self.particle_types} types."
        )
        logging.debug(
            f"Particle data arrays created. "
            f"Positions shape: {self.positions.shape}, "
            f"Velocities shape: {self.velocities.shape}, "
            f"Types shape: {self.types.shape}"
        )