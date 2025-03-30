import numpy as np
from sklearn.ensemble import RandomForestRegressor

class HousingSuitabilityModel:
    """
    A model for predicting housing development suitability based on geographical features.
    
    In a real application, this would be a trained machine learning model.
    For this example, we'll simulate the model behavior.
    """
    
    def __init__(self):
        """Initialize the housing suitability model."""
        # In a real application, this would load a pre-trained model
        self.model = None
        self.feature_importance = {
            'elevation': 0.15,
            'slope': 0.20,
            'distance_to_water': 0.12,
            'distance_to_roads': 0.18,
            'soil_quality': 0.15,
            'land_cover': 0.10,
            'flood_risk': 0.10
        }
    
    def predict(self, features):
        """
        Predict housing suitability based on geographical features.
        
        Args:
            features (dict): Dictionary of geographical features
            
        Returns:
            float: Suitability score (0-10)
            dict: Factor scores
        """
        # In a real application, this would use the model to make predictions
        # For this example, we'll generate simulated predictions
        
        # Check if all required features are present
        required_features = list(self.feature_importance.keys())
        missing_features = [f for f in required_features if f not in features]
        
        if missing_features:
            # If features are missing, return low suitability
            overall_score = 3.0
            factor_scores = {f: 3.0 for f in required_features}
            
            # Adjust scores for provided features
            for f in features:
                if f in required_features:
                    factor_scores[f] = np.random.uniform(4.0, 7.0)
        else:
            # Generate random factor scores (in a real app, these would be model predictions)
            factor_scores = {}
            for feature in required_features:
                # Base score on feature value
                base_score = self._evaluate_feature(feature, features[feature])
                
                # Add some randomness to simulate model variance
                factor_scores[feature] = min(10.0, max(0.0, base_score + np.random.normal(0, 0.5)))
            
            # Calculate overall score as weighted average
            overall_score = sum(factor_scores[f] * self.feature_importance[f] for f in required_features)
        
        return overall_score, factor_scores
    
    def _evaluate_feature(self, feature_name, feature_value):
        """
        Evaluate a single feature's contribution to suitability.
        
        Args:
            feature_name (str): Name of the feature
            feature_value: Value of the feature
            
        Returns:
            float: Feature score (0-10)
        """
        # This is a simplified evaluation - in a real model, this would be more sophisticated
        if feature_name == 'elevation':
            # Moderate elevations are better
            if 100 <= feature_value <= 500:
                return 8.0
            elif feature_value < 100 or feature_value > 1000:
                return 4.0
            else:
                return 6.0
                
        elif feature_name == 'slope':
            # Flatter areas are better
            if feature_value < 5:
                return 9.0
            elif feature_value < 15:
                return 6.0
            else:
                return 3.0
                
        elif feature_name == 'distance_to_water':
            # Closer to water is better, but not too close (flood risk)
            if 200 <= feature_value <= 2000:
                return 8.0
            elif feature_value < 200:
                return 5.0
            else:
                return 6.0
                
        elif feature_name == 'distance_to_roads':
            # Closer to roads is better
            if feature_value < 1000:
                return 9.0
            elif feature_value < 5000:
                return 6.0
            else:
                return 3.0
                
        elif feature_name == 'soil_quality':
            # Higher soil quality is better
            return feature_value
                
        elif feature_name == 'land_cover':
            # Certain land covers are better (e.g., grassland vs. forest)
            land_cover_scores = {
                'grass': 9.0,
                'shrub': 7.0,
                'sparse_forest': 5.0,
                'dense_forest': 3.0,
                'wetland': 2.0,
                'water': 0.0,
                'developed': 6.0,
                'barren': 8.0,
                'agriculture': 7.0
            }
            return land_cover_scores.get(feature_value, 5.0)
                
        elif feature_name == 'flood_risk':
            # Lower flood risk is better
            if feature_value == 'none':
                return 10.0
            elif feature_value == 'low':
                return 8.0
            elif feature_value == 'medium':
                return 4.0
            elif feature_value == 'high':
                return 1.0
            else:
                return 5.0
                
        else:
            # Default for unknown features
            return 5.0
