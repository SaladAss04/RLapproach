import sys
import os
# Set working directory to the root of the project
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

print(os.getcwd())
# List all files in the current directory
print(os.listdir())


import pytest
import numpy as np
from envs.aircraft import Aircraft, AircraftConfig


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_aircraft_initialization():
    """Test basic aircraft initialization and state."""
    aircraft = Aircraft(
        initial_position=(10, 10),
        initial_altitude=5000,
        initial_heading=90,
        initial_speed=200
    )
    
    assert aircraft.x == 10
    assert aircraft.y == 10
    assert aircraft.altitude == 5000
    assert aircraft.heading == 90
    assert aircraft.speed == 200
    assert len(aircraft.position_history) == 0

def test_aircraft_speed_limits():
    """Test aircraft speed constraints."""
    config = AircraftConfig(min_speed=100, max_speed=300)
    aircraft = Aircraft(
        initial_position=(0, 0),
        initial_altitude=5000,
        initial_heading=0,
        initial_speed=200,
        config=config
    )
    
    # Test speed increase within limits
    aircraft.update_speed(250)
    assert aircraft.speed < 250  # Should be limited by acceleration
    
    # Test speed decrease within limits
    aircraft.update_speed(150)
    assert aircraft.speed > 150  # Should be limited by deceleration
    
    # Test invalid speeds
    with pytest.raises(ValueError):
        aircraft.update_speed(50)  # Below minimum
    with pytest.raises(ValueError):
        aircraft.update_speed(350)  # Above maximum

def test_aircraft_altitude_changes():
    """Test aircraft climb and descent rates."""
    aircraft = Aircraft(
        initial_position=(0, 0),
        initial_altitude=5000,
        initial_heading=0,
        initial_speed=200
    )
    
    # Test climb
    initial_alt = aircraft.altitude
    aircraft.update_altitude(initial_alt + 1000)
    assert aircraft.altitude < initial_alt + 1000  # Should be limited by climb rate
    
    # Test descent
    aircraft.update_altitude(initial_alt - 1000)
    assert aircraft.altitude > initial_alt - 1000  # Should be limited by descent rate

def test_aircraft_turn_rates():
    """Test aircraft turn rate limits."""
    aircraft = Aircraft(
        initial_position=(0, 0),
        initial_altitude=5000,
        initial_heading=0,
        initial_speed=200
    )
    
    # Test right turn
    aircraft.update_heading(45)
    assert 0 < aircraft.heading < 45  # Should be limited by turn rate
    
    # Test left turn
    aircraft.update_heading(315)  # -45 degrees
    assert aircraft.heading > 315 or aircraft.heading < 45  # Should be limited by turn rate

def test_aircraft_movement():
    """Test aircraft position updates based on heading and speed."""
    aircraft = Aircraft(
        initial_position=(0, 0),
        initial_altitude=5000,
        initial_heading=90,  # Moving east
        initial_speed=120,  # 2 nautical miles per minute
        timestep=60  # 1 minute timestep
    )
    
    aircraft.step()
    assert pytest.approx(aircraft.x, rel=1e-2) == 2.0  # Should move 2nm east
    assert pytest.approx(aircraft.y, rel=1e-2) == 0.0  # Should not move north/south
    assert len(aircraft.position_history) == 1

def test_position_history():
    """Test aircraft position history tracking."""
    aircraft = Aircraft(
        initial_position=(0, 0),
        initial_altitude=5000,
        initial_heading=0,
        initial_speed=200
    )
    
    for _ in range(3):
        aircraft.step()
    
    assert len(aircraft.position_history) == 3
    assert isinstance(aircraft.position_history[0], tuple)
    assert len(aircraft.position_history[0]) == 2