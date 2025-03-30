import numpy as np
from sklearn.linear_model import LinearRegression

class PopulationEstimationModel:
    """
    A model for estimating population based on area characteristics.
    
    In a real application, this would be a trained machine learning model.
    For this example, we'll simulate the model behavior.
    """
    
    def __init__(self):
        """Initialize the population estimation model."""
        # In a real application, this would load a pre-trained model
        self.model = None
    
    def predict(self, area_size, housing_density, features=None):
        """
        Predict population based on area size, housing density, and optional features.
        
        Args:
            area_size (float): Area size in square kilometers
            housing_density (float): Housing units per square kilometer
            features (dict, optional): Additional area features
            
        Returns:
            dict: Population estimate data
        """
        # Calculate base housing units
        housing_units = area_size * housing_density
        
        # Apply adjustment factors based on features
        adjustment = 1.0
        
        if features:
            # Adjust based on urbanization
            if 'urbanization' in features:
                if features['urbanization'] == 'urban':
                    adjustment *= 1.05
                elif features['urbanization'] == 'suburban':
                    adjustment *= 1.0
                elif features['urbanization'] == 'rural':
                    adjustment *= 0.9
            
            # Adjust based on region
            if 'region' in features:
                region_factors = {
                    'northeast': 1.02,
                    'midwest': 1.0,
                    'south': 0.98,
                    'west': 1.03,
                    'other': 1.0
                }
                adjustment *= region_factors.get(features['region'], 1.0)
            
            # Adjust based on development type
            if 'development_type' in features:
                type_factors = {
                    'single_family': 1.0,
                    'multi_family': 1.2,
                    'mixed': 1.1,
                    'other': 1.0
                }
                adjustment *= type_factors.get(features['development_type'], 1.0)
        
        # Apply adjustment to housing units
        adjusted_housing_units = housing_units * adjustment
        
        # Calculate average household size (could be from model or input)
        avg_household_size = features.get('avg_household_size', 2.5) if features else 2.5
        
        # Calculate total population
        total_population = adjusted_housing_units * avg_household_size
        
        # Calculate population density
        population_density = total_population / area_size if area_size > 0 else 0
        
        # Generate age distribution
        # This would ideally come from a region-specific demographic model
        age_distribution = {
            'Under 18': 0.22,
            '18-24': 0.10,
            '25-34': 0.15,
            '35-44': 0.14,
            '45-54': 0.13,
            '55-64': 0.12,
            '65+': 0.14
        }
        
        # Adjust age distribution based on development type
        if features and 'development_type' in features:
            if features['development_type'] == 'retirement':
                # More older residents
                age_distribution = {
                    'Under 18': 0.05,
                    '18-24': 0.03,
                    '25-34': 0.05,
                    '35-44': 0.07,
                    '45-54': 0.15,
                    '55-64': 0.25,
                    '65+': 0.40
                }
            elif features['development_type'] == 'family':
                # More children and middle-aged adults
                age_distribution = {
                    'Under 18': 0.30,
                    '18-24': 0.08,
                    '25-34': 0.15,
                    '35-44': 0.20,
                    '45-54': 0.15,
                    '55-64': 0.07,
                    '65+': 0.05
                }
        
        demographic_breakdown = {
            age_group: round(total_population * percentage) 
            for age_group, percentage in age_distribution.items()
        }
        
        return {
            'area_size': area_size,
            'housing_units': adjusted_housing_units,
            'avg_household_size': avg_household_size,
            'total_population': total_population,
            'population_density': population_density,
            'demographic_breakdown': demographic_breakdown
        }
