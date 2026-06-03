import sys
import os
import pytest

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import map_fuel_type, map_transmission

def test_map_fuel_type():
    assert map_fuel_type('X') == 'Regular Gasoline'
    assert map_fuel_type('Z') == 'Premium Gasoline'
    assert map_fuel_type('D') == 'Diesel'
    assert map_fuel_type('E') == 'Ethanol (E85)'
    assert map_fuel_type('N') == 'Natural Gas (CNG)'
    assert map_fuel_type('UNKNOWN') == 'Other (UNKNOWN)'

def test_map_transmission():
    assert map_transmission('AS6') == 'Automatic (Select Shift)'
    assert map_transmission('AM7') == 'Automated Manual'
    assert map_transmission('AV') == 'Continuously Variable (CVT)'
    assert map_transmission('CVT') == 'Continuously Variable (CVT)'
    assert map_transmission('A8') == 'Automatic'
    assert map_transmission('M6') == 'Manual'
    assert map_transmission('XYZ') == 'Other'
