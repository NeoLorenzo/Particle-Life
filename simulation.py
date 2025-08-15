# simulation.py
"""
Handles the core simulation logic and physics calculations.

This module defines the Simulation class, which is responsible for
advancing the state of the particle system by one time step. It computes
inter-particle forces and updates their positions and velocities.
"""
import logging
import numpy as np
from typing import Dict, Any
from particle import ParticleSystem
from constants import WINDOW_WIDTH, WINDOW_HEIGHT

# --- Data Contracts ---
#
# class Simulation:
#   - __init__(self, particles: ParticleSystem, params: Dict[str, Any]):
#     - Inputs:
#       - particles: An initialized ParticleSystem object.
#       - params: Dictionary of simulation parameters from config.json.
#         - "friction": float
#         - "interaction_matrix": List[List[float]]
#         - "interaction_radius_min": float
#         - "interaction_radius_max": float
#         - "repulsion_strength": float
#     - Outputs: None
#     - Side Effects: Stores references to particles and parameters.
#       Converts interaction matrix to a NumPy array.
#
#   - step(self) -> None:
#     - Inputs: None (operates on internal state).
#     - Outputs: None
#     - Side Effects: Modifies the state of the internal ParticleSystem
#       object (positions and velocities).
#     - Invariants: Particle count remains constant. Particle positions
#       are clamped within the window bounds.

class Simulation:
    """
    Manages the simulation loop and physics calculations.
    """
    def __init__(self, particles: ParticleSystem, params: Dict[str, Any]):
        """
        Initializes the simulation environment.

        Args:
            particles (ParticleSystem): The particle system to simulate.
            params (Dict[str, Any]): Simulation parameters from config.
        """
        self.particles = particles
        self.friction = params['friction']
        self.interaction_matrix = np.array(params['interaction_matrix'])
        self.radius_min = params['interaction_radius_min']
        self.radius_max = params['interaction_radius_max']
        self.repulsion_strength = params.get('repulsion_strength', 1.0)
        
        logging.info("Simulation logic initialized.")

    def step(self):
        """
        Executes one time step of the simulation.
        """
        # 1. Calculate pairwise differences in positions
        delta_pos = self.particles.positions[:, np.newaxis, :] - self.particles.positions[np.newaxis, :, :]
        distances = np.linalg.norm(delta_pos, axis=2)
        direction = delta_pos / (distances[..., np.newaxis] + 1e-9)

        # Initialize total force array
        total_force = np.zeros_like(self.particles.positions)

        # 2. Calculate short-range repulsion force
        repulsion_mask = (distances > 0) & (distances < self.radius_min)
        repulsion_force = direction * self.repulsion_strength * (1 - distances / self.radius_min)[..., np.newaxis]
        total_force += np.sum(np.where(repulsion_mask[..., np.newaxis], repulsion_force, 0), axis=1)

        # 3. Calculate long-range interaction force
        interaction_mask = (distances >= self.radius_min) & (distances < self.radius_max)
        force_strength = self.interaction_matrix[self.particles.types[:, np.newaxis], self.particles.types[np.newaxis, :]]
        interaction_force = direction * force_strength[..., np.newaxis]
        total_force += np.sum(np.where(interaction_mask[..., np.newaxis], interaction_force, 0), axis=1)

        # 4. Update velocities with forces and friction
        self.particles.velocities += total_force
        self.particles.velocities *= (1.0 - self.friction)

        # 5. Update positions with velocities
        self.particles.positions += self.particles.velocities

        # 6. Handle boundary conditions (bouncing)
        pos = self.particles.positions
        vel = self.particles.velocities
        
        # Mask for particles outside horizontal bounds
        mask_x = (pos[:, 0] < 0) | (pos[:, 0] > WINDOW_WIDTH)
        vel[mask_x, 0] *= -1
        
        # Mask for particles outside vertical bounds
        mask_y = (pos[:, 1] < 0) | (pos[:, 1] > WINDOW_HEIGHT)
        vel[mask_y, 1] *= -1

        # Clamp positions to prevent particles from getting stuck outside bounds
        np.clip(pos[:, 0], 0, WINDOW_WIDTH, out=pos[:, 0])
        np.clip(pos[:, 1], 0, WINDOW_HEIGHT, out=pos[:, 1])