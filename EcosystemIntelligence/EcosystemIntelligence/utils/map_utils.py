import folium
from folium.plugins import Draw, MeasureControl
import numpy as np

def create_map_with_drawing_tools(center=[40.7128, -74.0060], zoom=10):
    """
    Create an interactive map with drawing tools.
    
    Args:
        center (list): Center coordinates for the map [lat, lon]
        zoom (int): Initial zoom level
        
    Returns:
        folium.Map: Map with drawing tools enabled
    """
    # Create a map
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='OpenStreetMap'
    )
    
    # Add drawing tools
    draw = Draw(
        export=True,
        position='topleft',
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={
            'featureGroup': None
        }
    )
    
    draw.add_to(m)
    
    # Add measure control
    measure_control = MeasureControl(
        position='bottomleft',
        primary_length_unit='kilometers',
        secondary_length_unit='miles',
        primary_area_unit='square kilometers',
        secondary_area_unit='acres'
    )
    
    measure_control.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def extract_coordinates(geojson_data):
    """
    Extract coordinates from GeoJSON data.
    
    Args:
        geojson_data (dict): GeoJSON data from the drawing tool
        
    Returns:
        list: List of coordinates [(lat, lon), ...]
    """
    # In a real implementation, this would parse the GeoJSON data
    # from the drawing tool to extract the coordinates
    
    if not geojson_data:
        return None
    
    try:
        # Extract coordinates based on feature type
        feature_type = geojson_data['type']
        
        if feature_type == 'Polygon':
            # Return the outer ring of the polygon
            return geojson_data['coordinates'][0]
        
        elif feature_type == 'Rectangle':
            # Return the rectangle coordinates
            return geojson_data['coordinates'][0]
        
        else:
            return None
    
    except (KeyError, IndexError):
        return None

def calculate_area(coordinates):
    """
    Calculate the area of a polygon in square kilometers.
    
    Args:
        coordinates (list): List of coordinates [(lat, lon), ...]
        
    Returns:
        float: Area in square kilometers
    """
    # This is a simplified area calculation and doesn't account for Earth's curvature
    # For more accuracy, libraries like GeoPy or proper GIS calculations should be used
    
    if not coordinates or len(coordinates) < 3:
        return 0.0
    
    # Convert to numpy array for easier calculation
    coords = np.array(coordinates)
    
    # Simple polygon area calculation (shoelace formula)
    x = coords[:, 0]
    y = coords[:, 1]
    
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    # Convert to square kilometers (very rough approximation)
    # This would need to be replaced with proper distance calculations
    # that account for the Earth's curvature in a real application
    area_km2 = area * 111.32 * 111.32  # Rough conversion
    
    return area_km2
