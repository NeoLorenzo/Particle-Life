# visualization.py
"""
Handles the visualization of the particle simulation using Pygame.
"""
import logging
import pygame
import numpy as np
from particle import ParticleSystem
from constants import BACKGROUND_COLOR, DEFAULT_PARTICLE_RADIUS, FULLSCREEN, UI_PANEL_WIDTH
from typing import Tuple, Optional

# Forward reference for type hinting to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulation import Simulation


# --- Data Contracts ---
#
# class Visualizer:
#   - __init__(self, width: int, height: int, particle_types: int, colors: Optional[list] = None):
#     - Inputs:
#       - width: int, width of the display window.
#       - height: int, height of the display window.
#       - particle_types: int, the number of particle types.
#       - colors: Optional list of RGB color lists (e.g., [[255,0,0], ...])
#         from the configuration. If None, colors are generated.
#     - Outputs: None
#     - Side Effects: Initializes Pygame and creates a display surface.
#
#   - draw(self, particles: ParticleSystem, simulation: "Simulation") -> bool:
#     - Inputs:
#       - particles: The ParticleSystem object containing the current state.
#       - simulation: The Simulation object holding the interaction matrix.
#     - Outputs:
#       - bool: False if the user has quit, True otherwise.
#     - Side Effects: Renders particles and UI to the screen, handles Pygame events,
#       and can modify the simulation's interaction matrix based on user input.

class Visualizer:
    """
    Renders the particle system state and provides interactive UI elements.
    """
    def __init__(self, particle_types: int, colors: Optional[list] = None):
        """
        Initializes Pygame and the display window.
        """
        pygame.init()
        pygame.font.init()

        if FULLSCREEN:
            display_info = pygame.display.Info()
            width, height = display_info.current_w, display_info.current_h
            self.screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
        else:
            # Fallback to a fixed size if not fullscreen
            width, height = 1500 + UI_PANEL_WIDTH, 700
            self.screen = pygame.display.set_mode((width, height))

        # The simulation area is the total width minus the UI panel
        self.sim_width = width - UI_PANEL_WIDTH
        self.sim_height = height
        
        # Create a dedicated surface for the simulation area
        self.sim_surface = pygame.Surface((self.sim_width, self.sim_height))

        pygame.display.set_caption("Particle Life")
        self.clock = pygame.time.Clock()
        
        # Load or generate colors for each particle type
        self.colors = self._initialize_colors(particle_types, colors)

        # --- UI Configuration (Rule 1: Application Constants) ---
        self.font = pygame.font.SysFont("monospace", 12)
        self.label_margin = 20 # Space for the colored circle labels
        # Position the matrix inside the new UI panel
        self.matrix_pos = (self.sim_width + 30, 10 + self.label_margin)
        self.cell_size = 40
        self.cell_padding = 2
        self.label_circle_radius = 8
        self.hovered_cell: Optional[Tuple[int, int]] = None
        self.scroll_sensitivity = 0.05

        # --- Reset Button Configuration ---
        matrix_pixel_width = particle_types * (self.cell_size + self.cell_padding) - self.cell_padding
        matrix_pixel_height = particle_types * (self.cell_size + self.cell_padding) - self.cell_padding
        button_y = self.matrix_pos[1] + matrix_pixel_height + 10
        self.reset_button_rect = pygame.Rect(self.matrix_pos[0], button_y, matrix_pixel_width, 30)
        
        # --- Randomize Button Configuration ---
        randomize_button_y = self.reset_button_rect.bottom + 5
        self.randomize_button_rect = pygame.Rect(self.matrix_pos[0], randomize_button_y, matrix_pixel_width, 30)

        self.button_color = (80, 80, 80)
        self.button_hover_color = (110, 110, 110)
        self.button_text_color = (255, 255, 255)

        logging.info(f"Visualizer initialized with Pygame display ({width}x{height}).")

    def _generate_colors(self, n: int, offset: int = 0) -> list:
        """Generates N visually distinct colors, with an optional offset for the seed."""
        return [pygame.Color(0).lerp(pygame.Color(((i + offset) * 997) % 256, ((i + offset) * 1337) % 256, ((i + offset) * 777) % 256), 0.7) for i in range(n)]

    def _initialize_colors(self, particle_types: int, config_colors: Optional[list]) -> list:
        """Initializes particle colors from config, falling back to procedural generation."""
        if not config_colors:
            logging.info(f"No colors found in config. Generating {particle_types} colors procedurally.")
            return self._generate_colors(particle_types)

        final_colors = []
        try:
            for rgb in config_colors:
                final_colors.append(pygame.Color(rgb))
        except (ValueError, TypeError) as e:
            logging.error(f"Could not parse colors from config due to invalid format: {e}. Falling back to procedural generation.")
            return self._generate_colors(particle_types)

        num_loaded = len(final_colors)
        if num_loaded < particle_types:
            num_to_generate = particle_types - num_loaded
            logging.warning(
                f"Config provides {num_loaded} colors, but {particle_types} are needed. "
                f"Generating the remaining {num_to_generate}."
            )
            final_colors.extend(self._generate_colors(num_to_generate, offset=num_loaded))
        elif num_loaded > particle_types:
            logging.warning(
                f"Config provides {num_loaded} colors, but only {particle_types} are needed. "
                "Ignoring excess colors."
            )
            final_colors = final_colors[:particle_types]
        else:
            logging.info(f"Successfully loaded {num_loaded} particle colors from configuration.")
            
        return final_colors

    def _get_matrix_cell_from_pos(self, pos: Tuple[int, int], matrix_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Converts a screen position to matrix cell coordinates if hovering over the matrix.
        """
        mx, my = self.matrix_pos
        rows, cols = matrix_shape
        for r in range(rows):
            for c in range(cols):
                cell_x = mx + c * (self.cell_size + self.cell_padding)
                cell_y = my + r * (self.cell_size + self.cell_padding)
                cell_rect = pygame.Rect(cell_x, cell_y, self.cell_size, self.cell_size)
                if cell_rect.collidepoint(pos):
                    return (r, c)
        return None

    def _draw_interaction_matrix(self, simulation: "Simulation"):
        """Renders the interaction matrix, its labels, and highlights the hovered cell."""
        matrix = simulation.interaction_matrix
        rows, cols = matrix.shape
        
        # --- Draw Labels ---
        for i in range(rows): # Row labels (left side)
            center_y = self.matrix_pos[1] + i * (self.cell_size + self.cell_padding) + self.cell_size / 2
            center_x = self.matrix_pos[0] - self.label_margin / 2
            pygame.draw.circle(self.screen, self.colors[i % len(self.colors)], (center_x, center_y), self.label_circle_radius)

        for i in range(cols): # Column labels (top side)
            center_x = self.matrix_pos[0] + i * (self.cell_size + self.cell_padding) + self.cell_size / 2
            center_y = self.matrix_pos[1] - self.label_margin / 2
            pygame.draw.circle(self.screen, self.colors[i % len(self.colors)], (center_x, center_y), self.label_circle_radius)

        # --- Draw Matrix Cells ---
        for r in range(rows):
            for c in range(cols):
                value = matrix[r, c]
                
                # Determine cell color based on value (Rule 3: Realism in behavior)
                # Green for attraction, Red for repulsion
                color_intensity = int(200 * abs(value))
                if value > 0:
                    bg_color = (0, color_intensity, 0) # Green
                elif value < 0:
                    bg_color = (color_intensity, 0, 0) # Red
                else:
                    bg_color = (50, 50, 50) # Gray for zero

                cell_x = self.matrix_pos[0] + c * (self.cell_size + self.cell_padding)
                cell_y = self.matrix_pos[1] + r * (self.cell_size + self.cell_padding)
                cell_rect = pygame.Rect(cell_x, cell_y, self.cell_size, self.cell_size)
                
                pygame.draw.rect(self.screen, bg_color, cell_rect)
                
                # Highlight hovered cell
                if self.hovered_cell == (r, c):
                    pygame.draw.rect(self.screen, (255, 255, 0), cell_rect, 2) # Yellow border

                # Render text
                text_surf = self.font.render(f"{value:.2f}", True, (255, 255, 255))
                text_rect = text_surf.get_rect(center=cell_rect.center)
                self.screen.blit(text_surf, text_rect)

    def _draw_reset_button(self, mouse_pos: Tuple[int, int]):
        """Draws the reset button and handles its hover state."""
        is_hovered = self.reset_button_rect.collidepoint(mouse_pos)
        color = self.button_hover_color if is_hovered else self.button_color
        
        pygame.draw.rect(self.screen, color, self.reset_button_rect, border_radius=5)
        
        text_surf = self.font.render("Reset", True, self.button_text_color)
        text_rect = text_surf.get_rect(center=self.reset_button_rect.center)
        self.screen.blit(text_surf, text_rect)

    def _draw_randomize_button(self, mouse_pos: Tuple[int, int]):
        """Draws the randomize button and handles its hover state."""
        is_hovered = self.randomize_button_rect.collidepoint(mouse_pos)
        color = self.button_hover_color if is_hovered else self.button_color
        
        pygame.draw.rect(self.screen, color, self.randomize_button_rect, border_radius=5)
        
        text_surf = self.font.render("Randomize", True, self.button_text_color)
        text_rect = text_surf.get_rect(center=self.randomize_button_rect.center)
        self.screen.blit(text_surf, text_rect)

    def draw(self, particles: ParticleSystem, simulation: "Simulation") -> bool:
        """
        Draws all particles and UI, and handles events.

        Returns:
            bool: False if the simulation should exit, True otherwise.
        """
        mouse_pos = pygame.mouse.get_pos()
        self.hovered_cell = self._get_matrix_cell_from_pos(mouse_pos, simulation.interaction_matrix.shape)

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("Quit event received. Shutting down visualizer.")
                return False
            
            # Add ESC key press to exit fullscreen mode
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    logging.info("ESC key pressed. Shutting down visualizer.")
                    return False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left mouse click
                    if self.reset_button_rect.collidepoint(mouse_pos):
                        simulation.interaction_matrix.fill(0.0)
                        logging.info("Interaction matrix reset to all zeros by user.")
                    elif self.randomize_button_rect.collidepoint(mouse_pos):
                        simulation.randomize_interaction_matrix()

            if event.type == pygame.MOUSEWHEEL:
                if self.hovered_cell:
                    r, c = self.hovered_cell
                    old_value = simulation.interaction_matrix[r, c]
                    
                    # event.y is 1 for scroll up, -1 for scroll down
                    change = event.y * self.scroll_sensitivity
                    new_value = np.clip(old_value + change, -1.0, 1.0)
                    
                    # Update the matrix in the simulation object directly
                    simulation.interaction_matrix[r, c] = new_value
                    
                    # Rule 2: Log the change
                    logging.info(
                        f"Interaction matrix updated at ({r}, {c}). "
                        f"Old: {old_value:.2f}, New: {new_value:.2f}"
                    )

        # Drawing
        # 1. Clear the main screen and the simulation surface
        self.screen.fill(BACKGROUND_COLOR)
        self.sim_surface.fill(BACKGROUND_COLOR)

        # 2. Draw particles onto the dedicated simulation surface
        draw_radius_check = simulation.radius_max

        for i in range(particles.particle_count):
            pos = particles.positions[i]
            p_type = particles.types[i]
            color = self.colors[p_type % len(self.colors)]

            # Determine necessary offsets for drawing ghosts (using sim dimensions)
            x_offsets = [0]
            if pos[0] < draw_radius_check:
                x_offsets.append(self.sim_width)
            elif pos[0] > self.sim_width - draw_radius_check:
                x_offsets.append(-self.sim_width)

            y_offsets = [0]
            if pos[1] < draw_radius_check:
                y_offsets.append(self.sim_height)
            elif pos[1] > self.sim_height - draw_radius_check:
                y_offsets.append(-self.sim_height)

            # Draw the particle and its ghosts onto the sim_surface
            for x_offset in x_offsets:
                for y_offset in y_offsets:
                    draw_pos = (int(pos[0] + x_offset), int(pos[1] + y_offset))
                    pygame.draw.circle(
                        self.sim_surface, # Draw on the sim surface
                        color,
                        draw_pos,
                        DEFAULT_PARTICLE_RADIUS
                    )
        
        # 3. Blit the simulation surface onto the main screen at (0, 0)
        self.screen.blit(self.sim_surface, (0, 0))

        # 4. Draw the UI directly onto the main screen in the panel area
        self._draw_interaction_matrix(simulation)
        self._draw_reset_button(mouse_pos)
        self._draw_randomize_button(mouse_pos)

        pygame.display.flip()
        return True

    def close(self):
        """Shuts down Pygame."""
        pygame.font.quit()
        pygame.quit()