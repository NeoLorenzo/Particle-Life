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
    Manages the simulation loop and physics calculations using a spatial grid
    for performance optimization.
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

        # Rule 7: Enforce data contracts. Validate config on initialization.
        num_types = self.particles.particle_types
        matrix_shape = self.interaction_matrix.shape
        if matrix_shape != (num_types, num_types):
            msg = (
                f"Configuration error: Interaction matrix shape {matrix_shape} "
                f"does not match particle_types ({num_types}). The matrix must be square "
                f"and its dimensions must equal the number of particle types."
            )
            logging.critical(msg)
            raise ValueError(msg)
        
        # --- Spatial Grid Initialization ---
        self.grid_cell_size = self.radius_max
        self.grid_width = int(np.ceil(WINDOW_WIDTH / self.grid_cell_size))
        self.grid_height = int(np.ceil(WINDOW_HEIGHT / self.grid_cell_size))
        self.grid = [[] for _ in range(self.grid_width * self.grid_height)]
        
        logging.info("Simulation logic initialized and configuration validated.")
        logging.info(
            f"Spatial grid enabled for performance: "
            f"{self.grid_width}x{self.grid_height} grid, "
            f"cell size {self.grid_cell_size:.2f}px."
        )

    def _update_grid(self):
        """Populates the spatial grid with current particle indices."""
        # Clear the grid
        for cell in self.grid:
            cell.clear()
        
        # Place particle indices into grid cells
        for i in range(self.particles.particle_count):
            pos = self.particles.positions[i]
            cell_x = int(pos[0] / self.grid_cell_size)
            cell_y = int(pos[1] / self.grid_cell_size)
            
            # Clamp to grid bounds
            cell_x = max(0, min(self.grid_width - 1, cell_x))
            cell_y = max(0, min(self.grid_height - 1, cell_y))

            self.grid[cell_x + cell_y * self.grid_width].append(i)

    def step(self):
        """
        Executes one time step of the simulation.
        """
        # 1. Update the spatial grid with particle locations
        self._update_grid()

        total_force = np.zeros_like(self.particles.positions)
        
        # 2. Calculate forces by checking neighboring grid cells
        for i in range(self.particles.particle_count):
            pos_i = self.particles.positions[i]
            type_i = self.particles.types[i]
            
            cell_x = int(pos_i[0] / self.grid_cell_size)
            cell_y = int(pos_i[1] / self.grid_cell_size)

            # Iterate over the 3x3 neighborhood of cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = cell_x + dx, cell_y + dy
                    
                    # Check if neighbor cell is within grid bounds
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        cell_idx = nx + ny * self.grid_width
                        for j in self.grid[cell_idx]:
                            if i == j:
                                continue

                            pos_j = self.particles.positions[j]
                            delta_pos = pos_j - pos_i
                            distance = np.linalg.norm(delta_pos)

                            if 0 < distance < self.radius_max:
                                direction = delta_pos / (distance + 1e-9)
                                
                                # Repulsion force (short-range)
                                if distance < self.radius_min:
                                    force = direction * self.repulsion_strength * (1 - distance / self.radius_min)
                                    total_force[i] -= force
                                # Interaction force (long-range)
                                else:
                                    type_j = self.particles.types[j]
                                    strength = self.interaction_matrix[type_i, type_j]
                                    force = direction * strength
                                    # REVERTED: Re-introduce original bug to match legacy behavior.
                                    # Positive interaction values should cause repulsion, not attraction.
                                    total_force[i] -= force

        # 3. Update velocities with forces and friction
        self.particles.velocities += total_force
        self.particles.velocities *= (1.0 - self.friction)

        # 4. Update positions with velocities
        self.particles.positions += self.particles.velocities

        # 5. Handle boundary conditions (bouncing)
        pos = self.particles.positions
        vel = self.particles.velocities
        
        mask_x = (pos[:, 0] < 0) | (pos[:, 0] > WINDOW_WIDTH)
        vel[mask_x, 0] *= -1
        
        mask_y = (pos[:, 1] < 0) | (pos[:, 1] > WINDOW_HEIGHT)
        vel[mask_y, 1] *= -1

        np.clip(pos[:, 0], 0, WINDOW_WIDTH, out=pos[:, 0])
        np.clip(pos[:, 1], 0, WINDOW_HEIGHT, out=pos[:, 1])