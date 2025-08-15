# visualization.py
"""
Handles the visualization of the particle simulation using Pygame.
"""
import logging
import pygame
import numpy as np
from particle import ParticleSystem
from constants import WINDOW_WIDTH, WINDOW_HEIGHT, BACKGROUND_COLOR, DEFAULT_PARTICLE_RADIUS

# --- Data Contracts ---
#
# class Visualizer:
#   - __init__(self, width: int, height: int):
#     - Inputs:
#       - width: int, width of the display window.
#       - height: int, height of the display window.
#     - Outputs: None
#     - Side Effects: Initializes Pygame and creates a display surface.
#
#   - draw(self, particles: ParticleSystem) -> bool:
#     - Inputs:
#       - particles: The ParticleSystem object containing the current state.
#     - Outputs:
#       - bool: False if the user has quit, True otherwise.
#     - Side Effects: Renders particles to the screen, handles Pygame events.

class Visualizer:
    """
    Renders the particle system state using Pygame.
    """
    def __init__(self, width: int, height: int):
        """
        Initializes Pygame and the display window.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Particle Life")
        self.clock = pygame.time.Clock()
        
        # Generate a color for each particle type
        self.colors = self._generate_colors(10) # Generate more than needed

        logging.info(f"Visualizer initialized with Pygame display ({width}x{height}).")

    def _generate_colors(self, n: int) -> list:
        """Generates N visually distinct colors."""
        return [pygame.Color(0).lerp(pygame.Color((i * 997) % 256, (i * 1337) % 256, (i * 777) % 256), 0.7) for i in range(n)]

    def draw(self, particles: ParticleSystem) -> bool:
        """
        Draws all particles and handles events.

        Returns:
            bool: False if the simulation should exit, True otherwise.
        """
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("Quit event received. Shutting down visualizer.")
                return False

        # Drawing
        self.screen.fill(BACKGROUND_COLOR)

        # Draw particles
        for i in range(particles.particle_count):
            pos = particles.positions[i]
            p_type = particles.types[i]
            color = self.colors[p_type % len(self.colors)]
            pygame.draw.circle(
                self.screen,
                color,
                (int(pos[0]), int(pos[1])),
                DEFAULT_PARTICLE_RADIUS
            )

        pygame.display.flip()
        return True

    def close(self):
        """Shuts down Pygame."""
        pygame.quit()