import numpy as np

class EnergyConsumptionModel:
    """
    A model for estimating energy consumption based on population and area characteristics.
    
    In a real application, this would be a trained machine learning model.
    For this example, we'll simulate the model behavior.
    """
    
    def __init__(self):
        """Initialize the energy consumption model."""
        # In a real application, this would load a pre-trained model
        self.model = None
        
        # Default per-capita consumption by region (kWh/year)
        self.region_consumption = {
            'northeast': 6500,
            'midwest': 7200,
            'south': 8000,
            'west': 6800,
            'other': 7200
        }
        
        # Monthly consumption distribution (percentage of annual)
        self.monthly_distribution = {
            'Jan': 0.11,
            'Feb': 0.10,
            'Mar': 0.09,
            'Apr': 0.08,
            'May': 0.07,
            'Jun': 0.07,
            'Jul': 0.08,
            'Aug': 0.08,
            'Sep': 0.07,
            'Oct': 0.08,
            'Nov': 0.08,
            'Dec': 0.09
        }
        
        # End-use distribution (percentage of total)
        self.end_use_distribution = {
            'Heating & Cooling': 0.42,
            'Water Heating': 0.18,
            'Lighting': 0.11,
            'Appliances': 0.14,
            'Electronics': 0.08,
            'Other': 0.07
        }
        
        # Emissions factors (kg CO2/kWh) by region
        self.emissions_factors = {
            'northeast': 0.28,
            'midwest': 0.45,
            'south': 0.40,
            'west': 0.25,
            'other': 0.35
        }
    
    def predict(self, population_data, features=None):
        """
        Predict energy consumption based on population and optional features.
        
        Args:
            population_data (dict): Population estimate data
            features (dict, optional): Additional area features
            
        Returns:
            dict: Energy consumption data
        """
        total_population = population_data['total_population']
        
        # Determine base per-capita consumption
        region = features.get('region', 'other') if features else 'other'
        base_consumption = self.region_consumption.get(region, self.region_consumption['other'])
        
        # Apply adjustment factors
        adjustment = 1.0
        
        if features:
            # Adjust based on development type
            if 'development_type' in features:
                type_factors = {
                    'single_family': 1.2,
                    'multi_family': 0.8,
                    'mixed': 1.0,
                    'other': 1.0
                }
                adjustment *= type_factors.get(features['development_type'], 1.0)
            
            # Adjust based on climate zone
            if 'climate_zone' in features:
                climate_factors = {
                    'hot': 1.2,  # Higher cooling needs
                    'cold': 1.3,  # Higher heating needs
                    'temperate': 0.9,  # Lower HVAC needs
                    'other': 1.0
                }
                adjustment *= climate_factors.get(features['climate_zone'], 1.0)
            
            # Adjust based on building efficiency
            if 'building_efficiency' in features:
                efficiency_factors = {
                    'low': 1.3,
                    'medium': 1.0,
                    'high': 0.7,
                    'other': 1.0
                }
                adjustment *= efficiency_factors.get(features['building_efficiency'], 1.0)
        
        # Calculate per-capita consumption
        per_capita = base_consumption * adjustment
        
        # Calculate total consumption
        total_consumption = total_population * per_capita
        
        # Calculate monthly consumption
        monthly_consumption = {
            month: total_consumption * factor 
            for month, factor in self.monthly_distribution.items()
        }
        
        # Calculate end-use consumption
        end_use_consumption = {
            use: total_consumption * factor 
            for use, factor in self.end_use_distribution.items()
        }
        
        # Calculate peak load (simplified)
        # Assuming peak demand occurs during 5% of the time with a 2.5 load factor
        peak_load = (total_consumption / (365 * 24)) * 2.5
        
        # Calculate CO2 emissions
        emissions_factor = self.emissions_factors.get(region, self.emissions_factors['other'])
        co2_emissions = total_consumption * emissions_factor / 1000  # tonnes of CO2
        
        return {
            'total_consumption': total_consumption,
            'per_capita': per_capita,
            'peak_load': peak_load,
            'monthly_consumption': monthly_consumption,
            'end_use_consumption': end_use_consumption,
            'co2_emissions': co2_emissions
        }

class RenewableEnergyModel:
    """
    A model for recommending renewable energy sources based on geographical features.
    
    In a real application, this would be a trained machine learning model.
    For this example, we'll simulate the model behavior.
    """
    
    def __init__(self):
        """Initialize the renewable energy recommendation model."""
        # In a real application, this would load a pre-trained model
        self.model = None
    
    def predict(self, geographical_features, energy_consumption):
        """
        Predict renewable energy potential based on geographical features.
        
        Args:
            geographical_features (dict): Geographical features
            energy_consumption (dict): Energy consumption data
            
        Returns:
            dict: Renewable energy recommendations
        """
        # Extract key features
        solar_irradiance = geographical_features.get('solar_irradiance', 4.0)
        wind_speed = geographical_features.get('wind_speed', 5.0)
        elevation_diff = geographical_features.get('elevation_diff', 100)
        water_bodies = geographical_features.get('water_bodies', 'None')
        geothermal_potential = geographical_features.get('geothermal_potential', 'Low')
        biomass_availability = geographical_features.get('biomass_availability', 'Low')
        
        # Calculate potential scores for each source
        solar_score = self._evaluate_solar(solar_irradiance)
        wind_score = self._evaluate_wind(wind_speed)
        hydro_score = self._evaluate_hydro(elevation_diff, water_bodies)
        geothermal_score = self._evaluate_geothermal(geothermal_potential)
        biomass_score = self._evaluate_biomass(biomass_availability)
        
        # Compile source data
        sources = {
            'Solar PV': {
                'score': solar_score,
                'potential_kwh': energy_consumption['total_consumption'] * (solar_score / 10) * 0.8,
                'color': 'rgba(255, 215, 0, 0.7)'
            },
            'Wind': {
                'score': wind_score,
                'potential_kwh': energy_consumption['total_consumption'] * (wind_score / 10) * 0.7,
                'color': 'rgba(135, 206, 250, 0.7)'
            },
            'Hydropower': {
                'score': hydro_score,
                'potential_kwh': energy_consumption['total_consumption'] * (hydro_score / 10) * 0.9,
                'color': 'rgba(70, 130, 180, 0.7)'
            },
            'Geothermal': {
                'score': geothermal_score,
                'potential_kwh': energy_consumption['total_consumption'] * (geothermal_score / 10) * 0.85,
                'color': 'rgba(178, 34, 34, 0.7)'
            },
            'Biomass': {
                'score': biomass_score,
                'potential_kwh': energy_consumption['total_consumption'] * (biomass_score / 10) * 0.75,
                'color': 'rgba(34, 139, 34, 0.7)'
            }
        }
        
        # Determine optimal energy mix
        total_score = sum(src['score'] for src in sources.values())
        
        if total_score > 0:
            energy_mix = {
                source: round((sources[source]['score'] / total_score) * 100)
                for source in sources
            }
        else:
            # Fallback if all scores are zero
            energy_mix = {source: 20 for source in sources}
        
        # Generate detailed recommendations
        for source in sources:
            sources[source]['recommendation'] = self._generate_recommendation(
                source, sources[source]['score'], energy_consumption
            )
            sources[source]['considerations'] = self._generate_considerations(
                source, sources[source]['score'], geographical_features
            )
        
        # Generate summary
        summary = self._generate_summary(sources, energy_consumption)
        
        return {
            'sources': sources,
            'energy_mix': energy_mix,
            'summary': summary,
            'total_potential': sum(src['potential_kwh'] for src in sources.values())
        }
    
    def _evaluate_solar(self, irradiance):
        """Evaluate solar energy potential."""
        if irradiance < 2:
            return max(1.0, irradiance)
        elif irradiance > 6:
            return min(10.0, 9.0 + (irradiance - 6) * 0.2)
        else:
            return 2.0 + (irradiance - 2) * (7.0 / 4.0)
    
    def _evaluate_wind(self, wind_speed):
        """Evaluate wind energy potential."""
        if wind_speed < 3:
            return max(0.5, wind_speed / 2)
        elif wind_speed > 10:
            return min(10.0, 9.0 + (wind_speed - 10) * 0.1)
        else:
            return 1.5 + (wind_speed - 3) * (7.5 / 7.0)
    
    def _evaluate_hydro(self, elevation_diff, water_bodies):
        """Evaluate hydropower potential."""
        # Base score on elevation difference
        if elevation_diff < 30:
            elev_score = max(0.1, elevation_diff / 30)
        elif elevation_diff > 500:
            elev_score = min(10.0, 8.0 + (elevation_diff - 500) * 0.004)
        else:
            elev_score = 1.0 + (elevation_diff - 30) * (7.0 / 470.0)
        
        # Water body factor
        water_factor = {
            'None': 0.0,
            'Small Stream': 0.3,
            'River': 0.8,
            'Lake': 0.6,
            'Ocean': 0.4  # For tidal potential
        }
        
        # Calculate final score
        return elev_score * water_factor.get(water_bodies, 0.0)
    
    def _evaluate_geothermal(self, potential):
        """Evaluate geothermal energy potential."""
        potential_map = {
            'Low': 2.5,
            'Medium': 6.0,
            'High': 9.0
        }
        return potential_map.get(potential, 1.0)
    
    def _evaluate_biomass(self, availability):
        """Evaluate biomass energy potential."""
        availability_map = {
            'Low': 2.0,
            'Medium': 5.5,
            'High': 8.0
        }
        return availability_map.get(availability, 1.0)
    
    def _generate_recommendation(self, source, score, energy_consumption):
        """Generate a recommendation for a specific energy source."""
        total_consumption = energy_consumption['total_consumption']
        
        if score < 3:
            return f"{source} has low potential in this area and is not recommended as a primary source."
        elif score < 6:
            return f"{source} has moderate potential and could supply approximately {int(score * 10)}% of energy needs."
        else:
            return f"{source} has high potential and could be a primary energy source, potentially supplying {int(score * 10)}% of energy needs."
    
    def _generate_considerations(self, source, score, features):
        """Generate implementation considerations for a specific energy source."""
        considerations = []
        
        if source == 'Solar PV':
            considerations.append(f"Average solar irradiance: {features.get('solar_irradiance', 'N/A')} kWh/m²/day")
            if score > 7:
                considerations.append("Excellent conditions for both rooftop and ground-mounted systems")
            elif score > 4:
                considerations.append("Good conditions for rooftop solar, consider east-west orientation")
            else:
                considerations.append("Limited solar potential, focus on high-efficiency panels")
                
        elif source == 'Wind':
            considerations.append(f"Average wind speed: {features.get('wind_speed', 'N/A')} m/s")
            if score > 7:
                considerations.append("Excellent conditions for utility-scale or community wind")
            elif score > 4:
                considerations.append("Consider small to medium turbines in optimal locations")
            else:
                considerations.append("Limited wind resource, may not be economically viable")
                
        elif source == 'Hydropower':
            considerations.append(f"Elevation difference: {features.get('elevation_diff', 'N/A')} m")
            considerations.append(f"Water body type: {features.get('water_bodies', 'N/A')}")
            if score > 7:
                considerations.append("Strong potential for small-scale hydropower development")
            elif score > 4:
                considerations.append("Consider micro-hydro installations where water flow is consistent")
            else:
                considerations.append("Limited hydro potential, may require significant infrastructure")
                
        elif source == 'Geothermal':
            considerations.append(f"Geothermal potential: {features.get('geothermal_potential', 'N/A')}")
            if score > 7:
                considerations.append("Excellent conditions for both heating/cooling and possibly electricity")
            elif score > 4:
                considerations.append("Good potential for ground-source heat pumps for heating and cooling")
            else:
                considerations.append("Consider limited applications for ground-source heat exchange")
                
        elif source == 'Biomass':
            considerations.append(f"Biomass availability: {features.get('biomass_availability', 'N/A')}")
            if score > 7:
                considerations.append("Strong potential for biomass heating and possibly CHP")
            elif score > 4:
                considerations.append("Consider biomass for supplemental heating or district systems")
            else:
                considerations.append("Limited biomass resources, ensure sustainable sourcing")
        
        return considerations
    
    def _generate_summary(self, sources, energy_consumption):
        """Generate an overall summary of renewable energy recommendations."""
        # Identify top sources
        top_sources = sorted(sources.items(), key=lambda x: x[1]['score'], reverse=True)
        
        if top_sources[0][1]['score'] < 4:
            summary = """
            This area has limited renewable energy potential across all evaluated sources.
            Consider a diverse mix of sources combined with energy efficiency measures and grid connection.
            """
        elif top_sources[0][1]['score'] > 7:
            primary = top_sources[0][0]
            secondary = top_sources[1][0] if top_sources[1][1]['score'] > 4 else None
            
            if secondary:
                summary = f"""
                This area has excellent potential for {primary} as a primary energy source,
                complemented by {secondary} as a secondary source. Together, they could provide
                a significant portion of the development's energy needs.
                """
            else:
                summary = f"""
                This area has excellent potential for {primary} as a primary energy source,
                which could provide a significant portion of the development's energy needs.
                Other renewable sources have limited potential and should be considered as supplements.
                """
        else:
            # Moderate potential
            viable_sources = [s[0] for s in top_sources if s[1]['score'] > 4]
            
            if viable_sources:
                sources_text = ", ".join(viable_sources)
                summary = f"""
                This area has moderate potential for renewable energy development.
                A balanced mix of {sources_text} is recommended to provide
                diversity and resilience in the energy supply.
                """
            else:
                summary = """
                This area has limited renewable energy potential. Consider focusing on
                energy efficiency measures and selective implementation of the most viable
                renewable sources, likely requiring significant grid connection for reliability.
                """
        
        return summary.strip()
