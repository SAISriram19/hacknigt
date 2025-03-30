import numpy as np
import pandas as pd

def calculate_energy_consumption(population_data, avg_consumption, efficiency_factor):
    """
    Calculate the estimated energy consumption for the population.
    
    Args:
        population_data (dict): Population estimate data
        avg_consumption (float): Average annual energy consumption per person (kWh)
        efficiency_factor (float): Efficiency factor to adjust consumption
        
    Returns:
        dict: Energy consumption data
    """
    # Extract total population
    total_population = population_data['total_population']
    
    # Calculate total energy consumption
    total_consumption = total_population * avg_consumption * efficiency_factor
    
    # Calculate peak load (simplified)
    # In a real application, this would use load profiles and diversity factors
    peak_load = total_consumption / (365 * 24) * 2.5  # Rough peak factor of 2.5
    
    # Calculate monthly consumption (simplified seasonal variation)
    monthly_factor = {
        'Jan': 1.2, 'Feb': 1.15, 'Mar': 1.05, 'Apr': 0.95, 'May': 0.9, 
        'Jun': 0.95, 'Jul': 1.0, 'Aug': 1.05, 'Sep': 0.9, 
        'Oct': 0.95, 'Nov': 1.05, 'Dec': 1.2
    }
    
    monthly_consumption = {
        month: (total_consumption / 12) * factor 
        for month, factor in monthly_factor.items()
    }
    
    # End-use breakdown (simplified)
    end_use_breakdown = {
        'Heating & Cooling': 0.42,
        'Water Heating': 0.18,
        'Lighting': 0.11,
        'Appliances': 0.14,
        'Electronics': 0.08,
        'Other': 0.07
    }
    
    end_use_consumption = {
        use: total_consumption * percentage 
        for use, percentage in end_use_breakdown.items()
    }
    
    # Calculate CO2 emissions (simplified, based on average grid mix)
    # Using 0.45 kg CO2/kWh as a rough approximation
    co2_emissions = total_consumption * 0.45 / 1000  # tonnes of CO2
    
    return {
        'total_consumption': total_consumption,
        'peak_load': peak_load,
        'monthly_consumption': monthly_consumption,
        'end_use_consumption': end_use_consumption,
        'co2_emissions': co2_emissions
    }

def recommend_renewable_energy(coordinates, energy_consumption, geo_features):
    """
    Recommend renewable energy sources based on geographical features and energy consumption.
    
    Args:
        coordinates (list): List of coordinates defining the area
        energy_consumption (dict): Energy consumption data
        geo_features (dict): Geographical features relevant for renewable energy
        
    Returns:
        dict: Renewable energy recommendations
    """
    # Extract key parameters
    total_consumption = energy_consumption['total_consumption']
    peak_load = energy_consumption['peak_load']
    
    # Evaluate suitability of different renewable energy sources
    solar_score = evaluate_solar_potential(geo_features['solar_irradiance'])
    wind_score = evaluate_wind_potential(geo_features['wind_speed'])
    hydro_score = evaluate_hydro_potential(geo_features['elevation_diff'], geo_features['water_bodies'])
    geothermal_score = evaluate_geothermal_potential(geo_features['geothermal_potential'])
    biomass_score = evaluate_biomass_potential(geo_features['biomass_availability'])
    
    # Create detailed source information
    sources = {
        'Solar PV': {
            'score': solar_score,
            'color': 'rgba(255, 215, 0, 0.7)',
            'recommendation': f"Solar potential is {'high' if solar_score > 7 else 'moderate' if solar_score > 4 else 'low'}.",
            'considerations': [
                f"Average solar irradiance: {geo_features['solar_irradiance']} kWh/m²/day",
                f"Could provide approximately {int(solar_score * 10)}% of energy needs",
                "Consider rooftop and ground-mounted installations"
            ]
        },
        'Wind': {
            'score': wind_score,
            'color': 'rgba(135, 206, 250, 0.7)',
            'recommendation': f"Wind energy potential is {'high' if wind_score > 7 else 'moderate' if wind_score > 4 else 'low'}.",
            'considerations': [
                f"Average wind speed: {geo_features['wind_speed']} m/s",
                f"Could provide approximately {int(wind_score * 10)}% of energy needs",
                "Consider small-scale turbines integrated into the development"
            ]
        },
        'Hydropower': {
            'score': hydro_score,
            'color': 'rgba(70, 130, 180, 0.7)',
            'recommendation': f"Hydropower potential is {'high' if hydro_score > 7 else 'moderate' if hydro_score > 4 else 'low'}.",
            'considerations': [
                f"Elevation difference: {geo_features['elevation_diff']} m",
                f"Water body type: {geo_features['water_bodies']}",
                "Consider micro-hydro installations if suitable water sources exist"
            ]
        },
        'Geothermal': {
            'score': geothermal_score,
            'color': 'rgba(178, 34, 34, 0.7)',
            'recommendation': f"Geothermal potential is {'high' if geothermal_score > 7 else 'moderate' if geothermal_score > 4 else 'low'}.",
            'considerations': [
                f"Geothermal potential: {geo_features['geothermal_potential']}",
                "Consider ground-source heat pumps for heating and cooling",
                "Deep geothermal may require detailed geological surveys"
            ]
        },
        'Biomass': {
            'score': biomass_score,
            'color': 'rgba(34, 139, 34, 0.7)',
            'recommendation': f"Biomass potential is {'high' if biomass_score > 7 else 'moderate' if biomass_score > 4 else 'low'}.",
            'considerations': [
                f"Biomass availability: {geo_features['biomass_availability']}",
                "Consider combined heat and power (CHP) systems",
                "Evaluate sustainable sourcing of biomass materials"
            ]
        }
    }
    
    # Determine recommended energy mix based on scores
    total_score = sum(source['score'] for source in sources.values())
    
    if total_score > 0:
        energy_mix = {
            source: round((sources[source]['score'] / total_score) * 100)
            for source in sources
        }
    else:
        # Fallback if all scores are zero
        energy_mix = {source: 20 for source in sources}
    
    # Prepare summary
    scores_sorted = sorted(sources.items(), key=lambda x: x[1]['score'], reverse=True)
    top_sources = [source for source, data in scores_sorted[:2]]
    
    if scores_sorted[0][1]['score'] > 7:
        primary_recommendation = f"{top_sources[0]} is highly recommended as the primary energy source."
    elif scores_sorted[0][1]['score'] > 4:
        primary_recommendation = f"{top_sources[0]} is recommended as the primary energy source with supplementary sources."
    else:
        primary_recommendation = "A diverse mix of renewable sources is recommended as no single source has high potential."
    
    summary = f"""
    Based on the geographical features and energy requirements, {primary_recommendation}
    
    The recommended energy mix can supply approximately 
    {min(100, int(total_score * 10))}% of the development's energy needs.
    
    Additional grid connection may be required for reliability and peak demand periods.
    """
    
    # Implementation notes
    implementation_notes = [
        f"Total annual energy consumption: {total_consumption:,.0f} kWh",
        f"Peak load: {peak_load:.2f} MW",
        "Consider energy storage solutions to address intermittency",
        "Implement smart grid technologies for optimal resource management",
        "Explore community energy schemes and microgrids",
        "Integrate energy-efficient building design to reduce overall consumption"
    ]
    
    return {
        'sources': sources,
        'energy_mix': energy_mix,
        'summary': summary,
        'implementation_notes': implementation_notes
    }

def evaluate_solar_potential(solar_irradiance):
    """
    Evaluate solar energy potential based on irradiance.
    
    Args:
        solar_irradiance (float): Average solar irradiance in kWh/m²/day
        
    Returns:
        float: Suitability score (0-10)
    """
    # Solar potential increases with irradiance
    # Below 2 kWh/m²/day is generally considered poor
    # Above 5 kWh/m²/day is generally considered excellent
    
    if solar_irradiance < 2:
        return 2.0
    elif solar_irradiance > 5:
        return 9.0
    else:
        # Linear scaling between 2 and 5 kWh/m²/day
        return 2.0 + (solar_irradiance - 2) * (7.0 / 3.0)

def evaluate_wind_potential(wind_speed):
    """
    Evaluate wind energy potential based on average wind speed.
    
    Args:
        wind_speed (float): Average wind speed in m/s
        
    Returns:
        float: Suitability score (0-10)
    """
    # Wind potential increases with wind speed
    # Below 3 m/s is generally not viable
    # Above 8 m/s is excellent
    
    if wind_speed < 3:
        return max(0, wind_speed * 0.5)
    elif wind_speed > 8:
        return 9.0 + min(1.0, (wind_speed - 8) * 0.2)
    else:
        # Linear scaling between 3 and 8 m/s
        return 1.5 + (wind_speed - 3) * (7.5 / 5.0)

def evaluate_hydro_potential(elevation_diff, water_bodies):
    """
    Evaluate hydropower potential based on elevation difference and water bodies.
    
    Args:
        elevation_diff (float): Elevation difference in meters
        water_bodies (str): Type of water bodies present
        
    Returns:
        float: Suitability score (0-10)
    """
    # Base score on elevation difference
    if elevation_diff < 50:
        elev_score = elevation_diff / 10
    elif elevation_diff < 200:
        elev_score = 5.0 + (elevation_diff - 50) * (3.0 / 150)
    else:
        elev_score = 8.0 + min(2.0, (elevation_diff - 200) * 0.01)
    
    # Adjust based on water bodies
    water_factor = {
        'None': 0.0,
        'Small Stream': 0.4,
        'River': 0.8,
        'Lake': 0.7,
        'Ocean': 0.5  # Tidal potential
    }
    
    # Calculate final score
    score = elev_score * water_factor.get(water_bodies, 0.0)
    
    return score

def evaluate_geothermal_potential(geothermal_potential):
    """
    Evaluate geothermal energy potential.
    
    Args:
        geothermal_potential (str): Qualitative geothermal potential
        
    Returns:
        float: Suitability score (0-10)
    """
    # Map qualitative potential to score
    potential_map = {
        'Low': 3.0,
        'Medium': 6.0,
        'High': 9.0
    }
    
    # Get score from map with default
    return potential_map.get(geothermal_potential, 2.0)

def evaluate_biomass_potential(biomass_availability):
    """
    Evaluate biomass energy potential.
    
    Args:
        biomass_availability (str): Qualitative biomass availability
        
    Returns:
        float: Suitability score (0-10)
    """
    # Map qualitative availability to score
    availability_map = {
        'Low': 3.0,
        'Medium': 6.0,
        'High': 8.5
    }
    
    # Get score from map with default
    return availability_map.get(biomass_availability, 2.0)
