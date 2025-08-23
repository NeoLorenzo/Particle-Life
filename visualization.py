# visualization.py
"""
Handles the visualization of the particle simulation using Pygame.
"""
import logging
import pygame
import numpy as np
from particle import ParticleSystem
from constants import (
    BACKGROUND_COLOR, DEFAULT_PARTICLE_RADIUS, FULLSCREEN, UI_PANEL_WIDTH,
    MOTION_BLUR_ALPHA, PARTICLE_HALO_RATIO, PARTICLE_HALO_ALPHA,
    UI_BACKGROUND_ALPHA, VIBRANT_COLORS, VELOCITY_GLOW_MIN_ALPHA,
    VELOCITY_GLOW_MAX_ALPHA
)
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
    def __init__(self, particle_types: int, colors: Optional[list] = None, sim_params: Optional[dict] = None):
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
        # Create a surface for the motion blur effect. This surface will be blitted
        # onto the main simulation surface each frame to create fading trails.
        self.blur_surface = pygame.Surface((self.sim_width, self.sim_height), pygame.SRCALPHA)
        self.blur_surface.fill((BACKGROUND_COLOR[0], BACKGROUND_COLOR[1], BACKGROUND_COLOR[2], MOTION_BLUR_ALPHA))
        
        # Create a surface for the UI panel background
        self.ui_panel_surface = pygame.Surface((UI_PANEL_WIDTH, self.sim_height), pygame.SRCALPHA)
        self.ui_panel_surface.fill((40, 40, 40, UI_BACKGROUND_ALPHA))

        pygame.display.set_caption("Particle Life")
        self.clock = pygame.time.Clock()
        
        # Load or generate colors for each particle type
        self.colors = self._initialize_colors(particle_types, colors)

        # --- Pre-render Halos for Performance (Rule 11) ---
        self.halo_surfaces = self._pre_render_halos()

        # --- UI Configuration (Rule 1: Application Constants) ---
        # Use a cleaner, sans-serif font. Pygame will fall back if 'Segoe UI' is not found.
        try:
            self.font_title = pygame.font.SysFont("Segoe UI", 16, bold=True)
            self.font_main = pygame.font.SysFont("Segoe UI", 14)
            self.font_main_bold = pygame.font.SysFont("Segoe UI", 14, bold=True)
        except pygame.error:
            logging.warning("Segoe UI font not found, falling back to default sans-serif.")
            self.font_title = pygame.font.SysFont(None, 20, bold=True)
            self.font_main = pygame.font.SysFont(None, 18)
            self.font_main_bold = pygame.font.SysFont(None, 18, bold=True)

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

        # --- UI Color Palette ---
        self.button_color = (80, 80, 80)
        self.button_hover_color = (110, 110, 110)
        self.text_color_title = (255, 255, 255)
        self.text_color_key = (200, 200, 200)   # Brighter grey for keys
        self.text_color_value = (255, 255, 255) # Pure white for values
        self.param_box_color = (60, 60, 60, 160) # Slightly lighter for individual boxes
        self.param_box_spacing = 4 # Vertical pixels between each parameter box

        # Store simulation parameters for display
        self.sim_params = sim_params if sim_params is not None else {}

        logging.info(f"Visualizer initialized with Pygame display ({width}x{height}).")

    def _generate_colors(self, n: int, offset: int = 0) -> list:
        """Generates N visually distinct colors, with an optional offset for the seed."""
        return [pygame.Color(0).lerp(pygame.Color(((i + offset) * 997) % 256, ((i + offset) * 1337) % 256, ((i + offset) * 777) % 256), 0.7) for i in range(n)]

    def _initialize_colors(self, particle_types: int, config_colors: Optional[list]) -> list:
        """Initializes particle colors from config, falling back to a vibrant default palette."""
        def get_default_colors(n_types):
            return [pygame.Color(VIBRANT_COLORS[i % len(VIBRANT_COLORS)]) for i in range(n_types)]

        if not config_colors:
            logging.info(f"No colors found in config. Using vibrant default palette.")
            return get_default_colors(particle_types)

        final_colors = []
        try:
            for rgb in config_colors:
                final_colors.append(pygame.Color(rgb))
        except (ValueError, TypeError) as e:
            logging.error(f"Could not parse colors from config due to invalid format: {e}. Falling back to vibrant default palette.")
            return get_default_colors(particle_types)

        num_loaded = len(final_colors)
        if num_loaded < particle_types:
            num_to_generate = particle_types - num_loaded
            logging.warning(
                f"Config provides {num_loaded} colors, but {particle_types} are needed. "
                f"Generating the remaining {num_to_generate} using the default palette."
            )
            default_palette = get_default_colors(particle_types)
            final_colors.extend(default_palette[num_loaded:])
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
                text_surf = self.font_main.render(f"{value:.2f}", True, self.text_color_title)
                text_rect = text_surf.get_rect(center=cell_rect.center)
                self.screen.blit(text_surf, text_rect)

    def _draw_reset_button(self, mouse_pos: Tuple[int, int]):
        """Draws the reset button and handles its hover state."""
        is_hovered = self.reset_button_rect.collidepoint(mouse_pos)
        color = self.button_hover_color if is_hovered else self.button_color
        
        pygame.draw.rect(self.screen, color, self.reset_button_rect, border_radius=5)
        
        text_surf = self.font_main.render("Reset", True, self.text_color_title)
        text_rect = text_surf.get_rect(center=self.reset_button_rect.center)
        self.screen.blit(text_surf, text_rect)

    def _draw_randomize_button(self, mouse_pos: Tuple[int, int]):
        """Draws the randomize button and handles its hover state."""
        is_hovered = self.randomize_button_rect.collidepoint(mouse_pos)
        color = self.button_hover_color if is_hovered else self.button_color
        
        pygame.draw.rect(self.screen, color, self.randomize_button_rect, border_radius=5)
        
        text_surf = self.font_main.render("Randomize", True, self.text_color_title)
        text_rect = text_surf.get_rect(center=self.randomize_button_rect.center)
        self.screen.blit(text_surf, text_rect)

    def _draw_simulation_parameters(self):
        """Renders simulation parameters in a list of individual, transparent boxes."""
        # --- Name Mapping for Readability ---
        param_name_map = {
            "seed": "Seed",
            "particle_count": "Particle Count",
            "particle_types": "Particle Types",
            "max_velocity": "Max Velocity",
            "friction": "Friction",
            "repulsion_strength": "Repulsion Strength",
            "velocity_damping_threshold": "Velocity Damping",
            "delta_time": "Delta Time",
            "interaction_radius_min": "Min Interaction Radius",
            "interaction_radius_max": "Max Interaction Radius"
        }

        # --- Layout Configuration ---
        box_v_padding = 8 # Vertical padding inside each box
        line_height = self.font_main.get_linesize()
        key_value_gap = 20
        
        panel_x = self.matrix_pos[0]
        panel_width = self.reset_button_rect.width
        current_y = self.randomize_button_rect.bottom + 20
        
        # Define column widths and positions
        key_max_width = (panel_width - key_value_gap) / 2 - box_v_padding
        value_max_width = key_max_width
        key_column_right_x = panel_x + box_v_padding + key_max_width
        value_column_left_x = key_column_right_x + key_value_gap

        # --- Render each parameter in its own box ---
        for key, value in self.sim_params.items():
            if key == "interaction_matrix":
                continue
            
            display_key = param_name_map.get(key, key.replace('_', ' ').title())
            display_value = f"{value:.2f}" if isinstance(value, float) else str(value)
            
            key_surfs = self._render_text_wrapped(display_key, self.font_main_bold, key_max_width, self.text_color_key)
            value_surfs = self._render_text_wrapped(display_value, self.font_main, value_max_width, self.text_color_value)
            
            num_lines = max(len(key_surfs), len(value_surfs))
            entry_height = num_lines * line_height
            box_height = entry_height + (box_v_padding * 2)

            # Draw the background box for this specific parameter
            box_rect = pygame.Rect(panel_x, current_y, panel_width, box_height)
            pygame.draw.rect(self.screen, self.param_box_color, box_rect, border_radius=6)

            # --- Render the text on top of the box ---
            text_start_y = current_y + box_v_padding
            
            # Draw each line of the key
            line_y = text_start_y
            for surf in key_surfs:
                rect = surf.get_rect(topright=(key_column_right_x, line_y))
                self.screen.blit(surf, rect)
                line_y += line_height

            # Draw each line of the value
            line_y = text_start_y
            for surf in value_surfs:
                rect = surf.get_rect(topleft=(value_column_left_x, line_y))
                self.screen.blit(surf, rect)
                line_y += line_height
            
            # Advance current_y for the next parameter box
            current_y += box_height + self.param_box_spacing

    def _render_text_wrapped(
        self, text: str, font: pygame.font.Font, max_width: int, color: tuple
    ) -> list:
        """
        Renders text, wrapping it to a new line if it exceeds max_width.
        Returns a list of rendered surfaces, one for each line.
        """
        words = text.split(' ')
        lines = []
        current_line = ""

        for word in words:
            # Test if adding the new word exceeds the width
            test_line = f"{current_line} {word}".strip()
            if font.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                # Line is too long, finalize the previous line
                lines.append(current_line)
                # Start a new line with the current word
                current_line = word
        
        lines.append(current_line) # Add the last line

        # Render each line into a surface with anti-aliasing
        return [font.render(line, True, color) for line in lines if line]

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
        # 1. Apply motion blur: Draw the semi-transparent blur surface over the
        #    simulation area. This fades the previous frame, creating trails.
        #    This is the correct way to prevent the "dirty background" artifact.
        self.sim_surface.blit(self.blur_surface, (0, 0))

        # 2. Draw particles with halos onto the dedicated simulation surface
        draw_radius_check = simulation.radius_max
        halo_radius = int(DEFAULT_PARTICLE_RADIUS * PARTICLE_HALO_RATIO)
        max_vel_sq = simulation.max_velocity ** 2 # Pre-calculate for performance

        for i in range(particles.particle_count):
            pos = particles.positions[i]
            vel = particles.velocities[i]
            p_type = particles.types[i]
            
            color_index = p_type % len(self.colors)
            base_color = self.colors[color_index]
            halo_surf = self.halo_surfaces[color_index]

            # --- Calculate Velocity-Based Glow ---
            # Rule 11: Avoid sqrt in hot loops. Use squared magnitude.
            speed_sq = vel[0]**2 + vel[1]**2
            # Normalize speed from 0 to 1
            normalized_speed = min(speed_sq / max_vel_sq, 1.0)
            
            # Map normalized speed to the alpha range
            dynamic_alpha = VELOCITY_GLOW_MIN_ALPHA + (normalized_speed * (VELOCITY_GLOW_MAX_ALPHA - VELOCITY_GLOW_MIN_ALPHA))
            halo_surf.set_alpha(dynamic_alpha)

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
                    
                    # Blit the pre-rendered halo with its new dynamic alpha value.
                    self.sim_surface.blit(halo_surf, (draw_pos[0] - halo_radius, draw_pos[1] - halo_radius))

                    # Draw the core particle on top
                    pygame.draw.circle(
                        self.sim_surface,
                        base_color,
                        draw_pos,
                        DEFAULT_PARTICLE_RADIUS
                    )
        
        # 3. Blit the simulation surface onto the main screen at (0, 0)
        self.screen.blit(self.sim_surface, (0, 0))

        # 4. Draw the UI panel background and then the UI elements on top
        self.screen.blit(self.ui_panel_surface, (self.sim_width, 0))
        self._draw_interaction_matrix(simulation)
        self._draw_reset_button(mouse_pos)
        self._draw_randomize_button(mouse_pos)
        self._draw_simulation_parameters()

        pygame.display.flip()
        return True

    def _pre_render_halos(self) -> list:
        """
        Pre-renders halo surfaces for each particle color to improve performance.
        """
        logging.debug("Pre-rendering particle halo surfaces...")
        halo_radius = int(DEFAULT_PARTICLE_RADIUS * PARTICLE_HALO_RATIO)
        diameter = halo_radius * 2
        surfaces = []
        for color in self.colors:
            halo_surf = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
            halo_color = pygame.Color(color.r, color.g, color.b, PARTICLE_HALO_ALPHA)
            pygame.draw.circle(
                halo_surf,
                halo_color,
                (halo_radius, halo_radius),
                halo_radius
            )
            surfaces.append(halo_surf)
        logging.debug(f"Finished pre-rendering {len(surfaces)} halo surfaces.")
        return surfaces

    def close(self):
        """Shuts down Pygame."""
        pygame.font.quit()
        pygame.quit()