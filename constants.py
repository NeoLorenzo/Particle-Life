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