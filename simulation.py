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
from numba import jit
from numba.core import types
from numba.typed import List

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

@jit(nopython=True)
def _update_grid_numba(positions, grid, grid_width, grid_height, grid_cell_size, world_width, world_height):
    """
    Numba-jitted function to populate the spatial grid.

    This version includes placing "ghost" particles in cells on the opposite
    side of the grid if a particle is within interaction range of a boundary.
    This is critical for making toroidal physics work with the grid optimization.
    """
    # Clear the grid
    for cell in grid:
        cell.clear()
    
    particle_count = positions.shape[0]
    for i in range(particle_count):
        pos = positions[i]
        
        # --- Primary Cell Placement ---
        cell_x = int(pos[0] / grid_cell_size)
        cell_y = int(pos[1] / grid_cell_size)
        grid[cell_x + cell_y * grid_width].append(i)

        # --- Ghost Particle Placement for Toroidal Wrapping ---
        # A particle near an edge must be "mirrored" to the other side so that
        # particles across the boundary can see it in their local grid search.
        # The check distance is grid_cell_size, which equals radius_max.
        near_left = pos[0] < grid_cell_size
        near_right = pos[0] > world_width - grid_cell_size
        near_top = pos[1] < grid_cell_size
        near_bottom = pos[1] > world_height - grid_cell_size

        # Add to ghost cells on the opposite side of the grid
        if near_right:
            grid[(cell_x - grid_width + 1) + cell_y * grid_width].append(i)
        if near_left:
            grid[(cell_x + grid_width - 1) + cell_y * grid_width].append(i)
        if near_bottom:
            grid[cell_x + (cell_y - grid_height + 1) * grid_width].append(i)
        if near_top:
            grid[cell_x + (cell_y + grid_height - 1) * grid_width].append(i)

        # Handle corners by adding to diagonal ghost cells
        if near_right and near_bottom:
            grid[(cell_x - grid_width + 1) + (cell_y - grid_height + 1) * grid_width].append(i)
        if near_left and near_bottom:
            grid[(cell_x + grid_width - 1) + (cell_y - grid_height + 1) * grid_width].append(i)
        if near_right and near_top:
            grid[(cell_x - grid_width + 1) + (cell_y + grid_height - 1) * grid_width].append(i)
        if near_left and near_top:
            grid[(cell_x + grid_width - 1) + (cell_y + grid_height - 1) * grid_width].append(i)

@jit(nopython=True)
def _calculate_forces_numba(
    positions, types, grid, grid_width, grid_height, grid_cell_size,
    radius_min, radius_min_sq, radius_max, radius_max_sq, repulsion_strength, interaction_matrix,
    world_width, world_height
):
    """
    Numba-jitted function to calculate inter-particle forces.
    This function is kept separate from the class for Numba compatibility.
    
    This version breaks Newton's 3rd Law by calculating interactions
    for each particle independently, allowing for non-conservative forces.
    """
    particle_count = positions.shape[0]
    total_force = np.zeros_like(positions)
    half_width = world_width / 2
    half_height = world_height / 2

    for i in range(particle_count):
        pos_i = positions[i]
        type_i = types[i]
        
        cell_x = int(pos_i[0] / grid_cell_size)
        cell_y = int(pos_i[1] / grid_cell_size)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = cell_x + dx, cell_y + dy
                
                if 0 <= nx < grid_width and 0 <= ny < grid_height:
                    cell_idx = nx + ny * grid_width
                    for j in grid[cell_idx]:
                        if i == j:
                            continue

                        pos_j = positions[j]
                        delta_pos = pos_j - pos_i
                        
                        # --- Toroidal distance correction (Rule 3: Realism) ---
                        if delta_pos[0] > half_width: delta_pos[0] -= world_width
                        elif delta_pos[0] < -half_width: delta_pos[0] += world_width
                        if delta_pos[1] > half_height: delta_pos[1] -= world_height
                        elif delta_pos[1] < -half_height: delta_pos[1] += world_height

                        distance_sq = delta_pos[0]**2 + delta_pos[1]**2

                        if 0 < distance_sq < radius_max_sq:
                            distance = np.sqrt(distance_sq)
                            # Direction is FROM i TO j
                            direction = delta_pos / (distance + 1e-9)
                            
                            force = np.zeros(2, dtype=np.float32)
                            if distance_sq < radius_min_sq:
                                # Repulsion is universal and symmetrical
                                force = -direction * repulsion_strength * (1 - distance / radius_min)
                            else:
                                # Asymmetrical interaction force
                                # Rule 8: This is a scientifically-grounded abstraction.
                                # The force peaks at an ideal distance and falls off,
                                # modeling phenomena like optimal bond lengths in chemistry
                                # or personal space in biology.
                                type_j = types[j]
                                strength = interaction_matrix[type_i, type_j]
                                
                                # Define the "sweet spot" for attraction as the midpoint
                                ideal_dist = (radius_min + radius_max) / 2.0
                                
                                if distance < ideal_dist:
                                    # Between min radius and ideal distance, force ramps up
                                    force_magnitude = strength * (distance - radius_min) / (ideal_dist - radius_min)
                                else:
                                    # Between ideal distance and max radius, force ramps down
                                    force_magnitude = strength * (1.0 - (distance - ideal_dist) / (radius_max - ideal_dist))
                                
                                force = direction * force_magnitude
                            
                            # Apply the calculated force only to particle i
                            total_force[i] += force
    return total_force

class Simulation:
    """
    Manages the simulation loop and physics calculations using a spatial grid
    for performance optimization.
    """
    def randomize_interaction_matrix(self):
        """
        Replaces the current interaction matrix with random values between -1.0 and 1.0.
        """
        # Rule 7 (SRP): This class is responsible for the simulation state,
        # so it is the correct place to modify the matrix.
        num_types = self.interaction_matrix.shape[0]
        self.interaction_matrix = (
            np.random.rand(num_types, num_types).astype(np.float32) * 2.0 - 1.0
        )
        logging.info("Interaction matrix randomized by user.")

    def __init__(self, particles: ParticleSystem, params: Dict[str, Any]):
        """
        Initializes the simulation environment.

        Args:
            particles (ParticleSystem): The particle system to simulate.
            params (Dict[str, Any]): Simulation parameters from config.
        """
        self.particles = particles
        # Rule 11.6: Use float32 for performance.
        self.max_velocity = np.float32(params.get('max_velocity', 5.0))
        self.friction = np.float32(params.get('friction', 0.05))
        self.velocity_damping_threshold = np.float32(params.get('velocity_damping_threshold', 0.0))
        self.delta_time = np.float32(params.get('delta_time', 0.1))
        self.interaction_matrix = np.array(params['interaction_matrix'], dtype=np.float32)
        self.radius_min = np.float32(params['interaction_radius_min'])
        self.radius_max = np.float32(params['interaction_radius_max'])
        self.repulsion_strength = np.float32(params.get('repulsion_strength', 1.0))

        # Rule 11: Performance - Pre-calculate squared radii to avoid sqrt in hot loop
        self.radius_min_sq = self.radius_min ** 2
        self.radius_max_sq = self.radius_max ** 2

        # Store world dimensions for toroidal physics calculations
        self.world_width = np.float32(WINDOW_WIDTH)
        self.world_height = np.float32(WINDOW_HEIGHT)

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
        
        # Rule 11.4: Numba requires typed data structures for JIT compilation.
        # We use a Numba Typed List instead of a standard Python list of lists.
        self.grid = List([List.empty_list(types.int64) for _ in range(self.grid_width * self.grid_height)])

        logging.info("Simulation logic initialized and configuration validated.")
        logging.info(
            f"Spatial grid enabled for performance: "
            f"{self.grid_width}x{self.grid_height} grid, "
            f"cell size {self.grid_cell_size:.2f}px."
        )

    def step(self):
        """
        Executes one time step of the simulation.
        """
        # 1. Update the spatial grid with particle locations (using Numba)
        _update_grid_numba(
            self.particles.positions, self.grid,
            self.grid_width, self.grid_height, self.grid_cell_size,
            self.world_width, self.world_height
        )

        # 2. Calculate forces using the Numba-jitted function
        total_force = _calculate_forces_numba(
            self.particles.positions, self.particles.types, self.grid,
            self.grid_width, self.grid_height, self.grid_cell_size,
            self.radius_min, self.radius_min_sq, self.radius_max, self.radius_max_sq,
            self.repulsion_strength, self.interaction_matrix,
            self.world_width, self.world_height
        )

        # 3. Update velocities with forces, scaled by delta_time for stability
        self.particles.velocities += total_force * self.delta_time

        # 4. Apply friction
        self.particles.velocities *= (1.0 - self.friction)

        # 5. Apply velocity cap
        velocities = self.particles.velocities
        speed = np.linalg.norm(velocities, axis=1)
        # Identify particles moving too fast
        over_speed_mask = speed > self.max_velocity
        # For those particles, scale their velocity vector back to the max_velocity
        velocities[over_speed_mask] = (
            velocities[over_speed_mask] / speed[over_speed_mask, np.newaxis]
        ) * self.max_velocity

        # 6. Apply stiction/damping for very low velocities (Rule 8)
        # This prevents jittering in stable configurations by zeroing out
        # velocities below a certain threshold.
        if self.velocity_damping_threshold > 0:
            # Recalculate speed if it was changed by the velocity cap
            speed[over_speed_mask] = self.max_velocity
            below_threshold_mask = speed < self.velocity_damping_threshold
            velocities[below_threshold_mask] = 0.0

        # 7. Update positions with velocities, scaled by delta_time
        self.particles.positions += self.particles.velocities * self.delta_time

        # 8. Handle boundary conditions (toroidal wrap-around)
        pos = self.particles.positions
        
        # Wrap positions using the modulo operator for an infinite space effect
        pos[:, 0] %= WINDOW_WIDTH
        pos[:, 1] %= WINDOW_HEIGHT