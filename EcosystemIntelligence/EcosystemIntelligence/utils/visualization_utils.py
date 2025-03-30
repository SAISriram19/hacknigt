import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_suitability_plot(housing_analysis):
    """
    Create a radar chart visualizing housing suitability factors.
    
    Args:
        housing_analysis (dict): Housing suitability analysis results
        
    Returns:
        plotly.graph_objects.Figure: Radar chart of suitability factors
    """
    # Extract factor scores
    factor_scores = housing_analysis['factor_scores']
    
    # Create radar chart
    categories = list(factor_scores.keys())
    values = list(factor_scores.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Suitability Factors',
        line_color='rgba(0, 128, 128, 0.8)',
        fillcolor='rgba(0, 128, 128, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        title="Housing Development Suitability Analysis",
        showlegend=False,
        height=400
    )
    
    return fig

def create_population_plot(population_data):
    """
    Create a chart visualizing population demographics.
    
    Args:
        population_data (dict): Population estimate data
        
    Returns:
        plotly.graph_objects.Figure: Population visualization
    """
    # Extract demographic breakdown
    demographics = population_data.get('demographic_breakdown', {})
    
    # Create bar chart of demographics
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(demographics.keys()),
        y=list(demographics.values()),
        marker_color='rgba(65, 105, 225, 0.7)'
    ))
    
    fig.update_layout(
        title="Estimated Population by Age Group",
        xaxis_title="Age Group",
        yaxis_title="Population",
        height=400
    )
    
    return fig

def create_energy_plot(energy_data):
    """
    Create a chart visualizing energy consumption.
    
    Args:
        energy_data (dict): Energy consumption data
        
    Returns:
        plotly.graph_objects.Figure: Energy consumption visualization
    """
    # Extract monthly consumption
    monthly = energy_data.get('monthly_consumption', {})
    
    # Create line chart of monthly consumption
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(monthly.keys()),
        y=list(monthly.values()),
        marker_color='rgba(255, 127, 80, 0.7)'
    ))
    
    fig.update_layout(
        title="Estimated Monthly Energy Consumption",
        xaxis_title="Month",
        yaxis_title="Energy Consumption (kWh)",
        height=400
    )
    
    return fig

def create_renewable_energy_plot(renewable_data):
    """
    Create a chart visualizing renewable energy potential.
    
    Args:
        renewable_data (dict): Renewable energy recommendation data
        
    Returns:
        plotly.graph_objects.Figure: Renewable energy visualization
    """
    # Extract source scores
    sources = renewable_data.get('sources', {})
    
    source_names = list(sources.keys())
    source_scores = [sources[name]['score'] for name in source_names]
    source_colors = [sources[name]['color'] for name in source_names]
    
    # Create bar chart of source scores
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=source_names,
        y=source_scores,
        marker_color=source_colors
    ))
    
    fig.update_layout(
        title="Renewable Energy Potential by Source",
        xaxis_title="Energy Source",
        yaxis_title="Suitability Score (0-10)",
        yaxis=dict(range=[0, 10]),
        height=400
    )
    
    return fig
