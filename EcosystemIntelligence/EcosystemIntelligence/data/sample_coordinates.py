def get_sample_coordinates():
    """
    Returns a sample set of coordinates that define a polygon.
    These could be used for testing or demonstration purposes when real drawing data is not available.
    
    Returns:
        list: List of [latitude, longitude] coordinates
    """
    # Sample coordinates for a polygon (roughly a small area in Central Park, NYC)
    return [
        [40.7812, -73.9665],
        [40.7812, -73.9645],
        [40.7792, -73.9645],
        [40.7792, -73.9665],
        [40.7812, -73.9665]  # Closing the polygon
    ]

def get_sample_features():
    """
    Returns a sample set of geographical features for a location.
    These could be used for testing or demonstration purposes.
    
    Returns:
        dict: Dictionary of geographical features
    """
    # Sample geographical features
    return {
        'elevation': 50,              # meters above sea level
        'slope': 3,                   # percentage
        'distance_to_water': 500,     # meters
        'distance_to_roads': 200,     # meters
        'soil_quality': 7,            # scale of 0-10
        'land_cover': 'grass',        # land cover type
        'flood_risk': 'low',          # flood risk category
        'solar_irradiance': 4.5,      # kWh/m²/day
        'wind_speed': 5.0,            # m/s at 10m height
        'elevation_diff': 20,         # meters
        'water_bodies': 'Small Stream', # type of water bodies
        'geothermal_potential': 'Medium', # geothermal potential category
        'biomass_availability': 'Low'   # biomass resource availability
    }
