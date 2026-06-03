import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def set_viz_style() -> None:
    """Sets the premium visualization style for the project plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'figure.facecolor': '#fafafa',
        'axes.facecolor': '#ffffff'
    })

def map_fuel_type(fuel: str) -> str:
    """
    Maps fuel type codes to descriptive names.

    Args:
        fuel: The fuel type code (e.g., 'X', 'Z').

    Returns:
        The descriptive fuel type name.
    """
    fuel_map = {
        'X': 'Regular Gasoline',
        'Z': 'Premium Gasoline',
        'D': 'Diesel',
        'E': 'Ethanol (E85)',
        'N': 'Natural Gas (CNG)'
    }
    return fuel_map.get(fuel, f'Other ({fuel})')

def map_transmission(t: Any) -> str:
    """
    Maps transmission codes to descriptive categories.

    Args:
        t: The transmission code.

    Returns:
        A descriptive transmission category.
    """
    t_str = str(t).upper()
    if t_str.startswith('AS'):
        return 'Automatic (Select Shift)'
    elif t_str.startswith('AM'):
        return 'Automated Manual'
    elif t_str.startswith('AV') or t_str.startswith('CV'):
        return 'Continuously Variable (CVT)'
    elif t_str.startswith('A'):
        return 'Automatic'
    elif t_str.startswith('M'):
        return 'Manual'
    else:
        return 'Other'

def ensure_dir(path: str) -> None:
    """
    Ensures that a directory exists, creating it if necessary.

    Args:
        path: The directory path.
    """
    os.makedirs(path, exist_ok=True)
