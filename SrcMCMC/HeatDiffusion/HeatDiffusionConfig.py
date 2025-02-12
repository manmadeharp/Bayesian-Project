from dataclasses import dataclass
from typing import Callable, Optional, Tuple

@dataclass
class DirichletHeatConfig:
    """Configuration for heat equation with Dirichlet BCs"""

    L: float = 1.0  # Domain length
    T: float = 1.0  # Final time
    nx: int = 100  # Number of spatial points
    nt: int = 100  # Number of time points
    left_bc: Callable = lambda t: 0  # Left boundary condition
    right_bc: Callable = lambda t: 0  # Right boundary condition

@dataclass
class HeatCauchyConfig:
    """Configuration for heat equation Cauchy problem"""
    T: float = 1.0  # Final time 
    nx: int = 100   # Number of spatial points
    nt: int = 100   # Number of time points
    ic: Callable = lambda t: 0  # Initial condition

