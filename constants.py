# constants.py
"""
Application-level constants.

These values are static and do not change between simulation runs.
They are fundamental to the application's framework, such as rendering
properties, default window sizes, or core physics settings that are not
part of the experimental configuration.
"""

# Visualization settings
# Set to True to run in borderless fullscreen mode.
# Set to False to run in a fixed-size window (1800x700).
FULLSCREEN = True
UI_PANEL_WIDTH = 300
FPS = 60
BACKGROUND_COLOR = (24, 24, 24) # Dark Gray
DEFAULT_PARTICLE_RADIUS = 4

# --- Visual Appeal Enhancements ---
# Alpha value for the motion blur effect (0-255). Lower is a longer trail.
MOTION_BLUR_ALPHA = 45
# Ratio of the halo size to the particle radius. e.g., 2 means halo is 2x bigger.
PARTICLE_HALO_RATIO = 2
# Alpha value for the particle halo (0-255).
PARTICLE_HALO_ALPHA = 40
# Alpha for the UI panel background
UI_BACKGROUND_ALPHA = 100

# --- Velocity Glow Effect ---
# The minimum alpha for a halo (for stationary particles).
VELOCITY_GLOW_MIN_ALPHA = 15
# The maximum alpha for a halo (for particles at max_velocity).
VELOCITY_GLOW_MAX_ALPHA = 120


# A curated list of vibrant default colors for particles, used if the
# config file does not provide a color list.
VIBRANT_COLORS = [
    (255, 0, 102),   # Hot Pink
    (0, 255, 255),   # Cyan
    (255, 204, 0),   # Gold
    (0, 255, 102),   # Bright Green
    (204, 0, 255),   # Purple
    (255, 102, 0)    # Orange
]