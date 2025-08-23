# Particle Life: A High-Performance Simulation of Emergent Behavior

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up--to--date-brightgreen)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

## Table of Contents
- [Core Features](#core-features)
- [Scientific Background & Motivation](#scientific-background--motivation)
- [Architectural Deep Dive](#architectural-deep-dive)
- [Getting Started](#getting-started)
- [Configuration & Customization](#configuration--customization)
- [Interactive Controls](#interactive-controls)
- [Community & Support](#community--support)
- [License](#license)
- [Project Structure](#project-structure)

---

![Particle Life Simulation GIF 1](https://raw.githubusercontent.com/NeoLorenzo/Particle-Life/main/demos/particle-life-sim-gif-1.gif)

This project presents a sophisticated and high-performance implementation of a "Particle Life" simulation. It is not merely a visual toy, but a robust framework engineered to explore the principles of **emergent behavior**, where complex, life-like patterns arise from a simple set of underlying rules. The simulation is architected for performance, realism, and extensibility, featuring a real-time interactive UI, a powerful physics core optimized with Numba, and a strict, modular design philosophy.

---

## Watch the Simulation in Action

Experience the mesmerizing emergent behavior of the simulation in this high-resolution timelapse. Witness how simple particles, governed by basic rules of attraction and repulsion, self-organize into complex, evolving ecosystems.

[![Particle Life Timelapse](https://img.youtube.com/vi/gWvgj65EFWM/0.jpg)](https://www.youtube.com/watch?v=gWvgj65EFWM)

---

## Core Features

*   **High-Performance Core:** The simulation loop is written in pure NumPy and accelerated with **Numba's JIT compilation**, enabling thousands of particles to be simulated in real-time.
*   **Emergent Complexity:** The simulation is designed to break Newton's Third Law, allowing for non-conservative forces. This asymmetry is the key to the rich, dynamic, and unpredictable behaviors that emerge.
*   **Real-Time Interactivity:** A dynamic UI panel, rendered with Pygame, allows you to directly manipulate the laws of physics. Tweak the interaction matrix in real-time with your mouse wheel and instantly observe the impact on the particle ecosystem.
*   **Scientifically-Grounded Abstractions:** The physics model balances 1:1 realism with performance by using principled abstractions, such as a force curve that models optimal bond lengths and a velocity damping threshold to prevent low-energy jitter.
*   **Advanced Visuals:** Features a modern aesthetic with motion blur, velocity-based particle glows, and a clean UI, all designed to make the emergent patterns beautiful and easy to interpret.
*   **Deterministic & Reproducible:** Every simulation run is controlled by a master seed, ensuring that experiments are fully deterministic and scientifically reproducible.
*   **Robust & Modular Architecture:** Built on SOLID design principles, the codebase is highly modular, readable, and extensible. Each component (physics, rendering, state management) is decoupled and communicates through clear data contracts.

![Particle Life Simulation GIF 2](https://raw.githubusercontent.com/NeoLorenzo/Particle-Life/main/demos/particle-life-sim-gif-2.gif)

---

## Scientific Background & Motivation

This simulation is an exploration into the field of **Artificial Life (A-Life)** and **emergent systems**. The core principle is that complex, seemingly intelligent behavior can arise (emerge) from a large number of simple agents following a basic set of rules, without any centralized control.

Inspired by foundational concepts like **Conway's Game of Life** and **Craig Reynolds' Boids**, this project takes the idea a step further by introducing continuous space and asymmetrical forces. The violation of Newton's Third Law is the critical ingredient that allows for true locomotion and the formation of self-propelling "organisms," creating a far richer and more dynamic ecosystem than is possible with conservative forces alone.

---

## Architectural Deep Dive

This simulation was built upon a set of strict, professional-grade architectural rules that ensure its stability, performance, and maintainability.

### 1. Principled Configuration Management

The system avoids "magic numbers" by strictly separating configuration into two types:
*   **`constants.py`**: Defines static, application-level constants like rendering properties and UI colors. These are integral to the framework and do not change between experiments.
*   **`config.json`**: Defines the parameters for a specific simulation run. This includes the number of particles, the physics rules, the master seed, and more. This design allows researchers to define and execute different experiments without ever touching the core source code.

### 2. Professional Logging Framework

All runtime messages are handled by Python's `logging` module. There are no `print` statements in the core logic.
*   **Structured Output:** Logs include timestamps, module names, and severity levels, and are written to both the console and a rotating log file.
*   **Performance-Aware:** In hot loops, logging is throttled to prevent I/O from becoming a bottleneck.
*   **Intelligent Debugging:** In the event of a critical failure, the system is designed to log the last few debug messages from each module, providing a clear snapshot of the state leading up to the error.

### 3. The Physics Engine: How It Works

The simulation's complexity arises from a carefully designed physics engine.

![Particle Life Simulation GIF 3](https://raw.githubusercontent.com/NeoLorenzo/Particle-Life/main/demos/particle-life-sim-gif-3.gif)

*   **Asymmetrical Interactions:** The core of the emergent behavior lies in the `interaction_matrix`. The force that particle type `A` exerts on type `B` is not necessarily equal and opposite to the force `B` exerts on `A`. This violation of Newton's Third Law creates net forces on the system, allowing for locomotion, chasing, and other complex dynamics.
*   **Scientifically-Grounded Force Curve:** The force between two particles is not linear. It follows a curve designed to mimic real-world phenomena:
    1.  **Close-Range Repulsion:** If particles get too close (`< interaction_radius_min`), a strong, universal repulsion force pushes them apart, preventing collapse.
    2.  **Interaction Zone:** Between the minimum and maximum radius, the force is governed by the interaction matrix. The force ramps up from the minimum radius to an "ideal" distance at the midpoint, then ramps back down towards the maximum radius. This models concepts like optimal chemical bond lengths or personal space in biological systems.
*   **Spatial Grid Optimization:** To avoid a costly O(n²) calculation for particle interactions, the simulation space is divided into a grid. Each particle only checks for interactions with particles in its own and adjacent grid cells. This optimization is implemented in a Numba-jitted function for maximum performance.
*   **Toroidal Universe:** The simulation space has wrap-around boundaries (a torus). Particles exiting one side of the screen seamlessly reappear on the opposite side. The spatial grid is intelligently designed to handle this by placing "ghost" particles in cells across the boundary, ensuring interactions are calculated correctly across the edges of the world.

### 4. Performance & Optimization

Performance is not an afterthought; it is a core design principle.
*   **Vectorized Operations:** All particle state is stored in NumPy arrays. Physics calculations are fully vectorized, eliminating slow Python loops from the hot path.
*   **Just-In-Time Compilation:** The most computationally expensive functions—force calculation and spatial grid updates—are decorated with `@jit(nopython=True)` from the Numba library. This compiles the Python code down to highly optimized machine code on the first run.
*   **Profiling-Driven Development:** Changes to the core loop are guided by `cProfile` to identify and eliminate bottlenecks, not guesswork.
*   **Pre-computation:** Visual elements like particle halos are pre-rendered at startup to reduce rendering overhead during the main loop.

---

## Getting Started

### Prerequisites

You will need Python 3.11 and the following libraries:
*   `pygame`
*   `numpy`
*   `numba`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NeoLorenzo/Particle-Life.git
    cd Particle-Life
    ```

2.  **Install the required packages:**
    ```bash
    pip install pygame numpy numba
    ```

3.  **Run the simulation:**
    ```bash
    python main.py
    ```
    Press `ESC` to exit the simulation.

---

## Configuration & Customization

Create your own unique particle universes by editing `config.json`.

*   `"particle_count"`: The total number of particles in the simulation.
*   `"particle_types"`: The number of different particle "species".
*   `"friction"`: A value from 0 to 1 that determines how quickly particles lose momentum.
*   `"interaction_matrix"`: The heart of the simulation. This is a `particle_types` x `particle_types` matrix. The value at `matrix[i][j]` defines the force that type `i` particles exert on type `j` particles.
    *   **Positive values** cause attraction.
    *   **Negative values** cause repulsion.
*   `"interaction_radius_max"`: The maximum distance at which particles can interact.

## Interactive Controls

![Particle Life UI GIF](https://raw.githubusercontent.com/NeoLorenzo/Particle-Life/main/demos/particle-life-sim-gif-4.gif)

The UI panel on the right side of the screen provides real-time control over the simulation's physics.

*   **Modify Interactions:** Hover your mouse over any cell in the interaction matrix and use the **mouse wheel** to increase or decrease the force value. Observe the immediate effect on the particle behaviors.
*   **Randomize:** Click the **"Randomize"** button to generate a completely new interaction matrix, instantly creating a new and unpredictable ecosystem.
*   **Reset:** Click the **"Reset"** button to set all interaction forces to zero.

---

## Community & Support

Have a question, found a bug, or want to share a fascinating new configuration? The best way to get in touch is by opening an issue or starting a discussion on the GitHub repository.

*   **Bug Reports & Feature Requests:** Please use the [Issues tab](https://github.com/NeoLorenzo/Particle-Life/issues).
*   **General Questions & Showcase:** Please use the [Discussions tab](https://github.com/NeoLorenzo/Particle-Life/discussions).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

This license allows for broad freedom to use, modify, and distribute the software, including for commercial purposes, as long as the original copyright and license notice are included.

---

## Project Structure

```
Particle-Life/
│
├── gifs/                     # Showcase GIFs for the README
├── logs/                     # Output directory for log files
│
├── config.json               # Defines simulation parameters for a run
├── constants.py              # Defines application-level static constants
├── main.py                   # Main entry point and simulation orchestrator
├── particle.py               # Manages particle state in NumPy arrays
├── simulation.py             # Core physics logic and Numba-optimized functions
├── utils.py                  # Helper functions (e.g., logging setup)
├── visualization.py          # Pygame-based rendering and UI handling
└── requirements.txt          # Project dependencies
```