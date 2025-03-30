import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import base64
import io
from datetime import datetime

from utils.map_utils import create_map_with_drawing_tools, extract_coordinates
from utils.analysis_utils import analyze_area_suitability, estimate_population
from utils.energy_utils import calculate_energy_consumption, recommend_renewable_energy
from utils.visualization_utils import create_suitability_plot, create_population_plot, create_energy_plot

# Set page config
st.set_page_config(
    page_title="Smart Housing & Energy Development",
    page_icon="🏙️",
    layout="wide"
)

# Application title
st.title("Smart Housing & Energy Development Analysis")
st.write("""
This application analyzes selected geographical areas to recommend optimal housing 
development locations, estimate population and energy consumption, and suggest 
suitable renewable energy sources based on geographical features.
""")

# Sidebar for navigation and controls
st.sidebar.title("Analysis Controls")
analysis_stage = st.sidebar.radio(
    "Analysis Stage",
    ["Area Selection", "Housing Suitability", "Population & Energy", "Renewable Energy Recommendation", "Report"]
)

# Initialize session state variables if they don't exist
if 'selected_area' not in st.session_state:
    st.session_state.selected_area = None
if 'housing_analysis' not in st.session_state:
    st.session_state.housing_analysis = None
if 'population_estimate' not in st.session_state:
    st.session_state.population_estimate = None
if 'energy_consumption' not in st.session_state:
    st.session_state.energy_consumption = None
if 'renewable_recommendation' not in st.session_state:
    st.session_state.renewable_recommendation = None

# Area Selection Stage
if analysis_stage == "Area Selection":
    st.header("Select Area for Analysis")
    st.write("Use the drawing tools on the map to outline the area you want to analyze.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create interactive map with drawing tools
        m = create_map_with_drawing_tools()
        folium_static(m, width=800, height=600)
    
    with col2:
        st.write("Instructions:")
        st.write("1. Click on the rectangle or polygon tool in the map")
        st.write("2. Draw your area of interest")
        st.write("3. Click 'Extract Area Data' when ready")
        
        # Create two columns for the buttons
        col1, col2 = st.columns(2)
        
        if col1.button("Extract Area Data"):
            # In a real app, this would extract the drawn shape from the map
            # For this example, we'll use a predefined area
            from data.sample_coordinates import get_sample_coordinates
            st.session_state.selected_area = get_sample_coordinates()
            st.success("Area data extracted successfully!")
            st.write("Selected area coordinates:")
            st.write(st.session_state.selected_area)
        
        # Only show the proceed button if area is selected
        if st.session_state.selected_area is not None:
            if col2.button("Proceed to Housing Suitability Analysis"):
                analysis_stage = "Housing Suitability"
                st.rerun()

# Housing Suitability Analysis Stage
elif analysis_stage == "Housing Suitability":
    st.header("Housing Development Suitability Analysis")
    
    if st.session_state.selected_area is None:
        st.warning("Please select an area first.")
        if st.button("Go to Area Selection"):
            analysis_stage = "Area Selection"
            st.rerun()
    else:
        st.write("Analyzing the selected area for housing development suitability...")
        
        # Run housing suitability analysis
        if st.session_state.housing_analysis is None:
            with st.spinner("Running analysis..."):
                st.session_state.housing_analysis = analyze_area_suitability(st.session_state.selected_area)
        
        # Display results
        st.subheader("Suitability Results")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = create_suitability_plot(st.session_state.housing_analysis)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("Suitability Factors:")
            for factor, score in st.session_state.housing_analysis['factor_scores'].items():
                st.metric(factor, f"{score:.2f}/10")
            
            st.write(f"Overall Suitability Score: {st.session_state.housing_analysis['overall_score']:.2f}/10")
            st.write(st.session_state.housing_analysis['recommendation'])
            
        if st.button("Proceed to Population & Energy Analysis"):
            analysis_stage = "Population & Energy"
            st.rerun()

# Population and Energy Consumption Stage
elif analysis_stage == "Population & Energy":
    st.header("Population Estimate & Energy Consumption")
    
    if st.session_state.housing_analysis is None:
        st.warning("Please complete housing suitability analysis first.")
        if st.button("Go to Housing Suitability Analysis"):
            analysis_stage = "Housing Suitability"
            st.rerun()
    else:
        # Parameters for estimation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Population Parameters")
            housing_density = st.slider("Housing Density (units/km²)", 10, 1000, 100)
            avg_household_size = st.slider("Average Household Size", 1.0, 6.0, 2.5)
            
        with col2:
            st.subheader("Energy Parameters")
            avg_consumption = st.slider("Average Annual Energy Consumption (kWh/person)", 1000, 10000, 4500)
            efficiency_factor = st.slider("Energy Efficiency Factor", 0.5, 1.5, 1.0)
        
        # Calculate estimates
        if st.button("Calculate Estimates") or st.session_state.population_estimate is not None:
            with st.spinner("Calculating population and energy estimates..."):
                if st.session_state.population_estimate is None:
                    st.session_state.population_estimate = estimate_population(
                        st.session_state.selected_area, 
                        housing_density, 
                        avg_household_size
                    )
                
                if st.session_state.energy_consumption is None:
                    st.session_state.energy_consumption = calculate_energy_consumption(
                        st.session_state.population_estimate, 
                        avg_consumption, 
                        efficiency_factor
                    )
            
            # Display results
            st.subheader("Estimation Results")
            col1, col2 = st.columns(2)
            
            with col1:
                pop_fig = create_population_plot(st.session_state.population_estimate)
                st.plotly_chart(pop_fig, use_container_width=True)
                
                st.metric("Estimated Total Population", f"{st.session_state.population_estimate['total_population']:,.0f}")
                st.metric("Area Size", f"{st.session_state.population_estimate['area_size']:.2f} km²")
                st.metric("Population Density", f"{st.session_state.population_estimate['population_density']:.2f} people/km²")
            
            with col2:
                energy_fig = create_energy_plot(st.session_state.energy_consumption)
                st.plotly_chart(energy_fig, use_container_width=True)
                
                st.metric("Total Annual Energy Consumption", f"{st.session_state.energy_consumption['total_consumption']:,.0f} kWh")
                st.metric("Peak Load Estimate", f"{st.session_state.energy_consumption['peak_load']:.2f} MW")
                st.metric("CO2 Emissions (Grid Mix)", f"{st.session_state.energy_consumption['co2_emissions']:,.0f} tonnes/year")
            
            if st.button("Proceed to Renewable Energy Recommendations"):
                analysis_stage = "Renewable Energy Recommendation"
                st.rerun()

# Renewable Energy Recommendation Stage
elif analysis_stage == "Renewable Energy Recommendation":
    st.header("Renewable Energy Recommendations")
    
    if st.session_state.energy_consumption is None:
        st.warning("Please complete population and energy analysis first.")
        if st.button("Go to Population & Energy Analysis"):
            analysis_stage = "Population & Energy"
            st.rerun()
    else:
        # Parameters for renewable energy analysis
        st.subheader("Geographical Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_solar_irradiance = st.slider("Average Solar Irradiance (kWh/m²/day)", 1.0, 7.0, 4.5)
            avg_wind_speed = st.slider("Average Wind Speed (m/s)", 0.0, 15.0, 5.0)
            
        with col2:
            elevation_diff = st.slider("Elevation Difference (m)", 0, 1000, 150)
            water_bodies = st.selectbox("Significant Water Bodies", ["None", "Small Stream", "River", "Lake", "Ocean"])
            
        with col3:
            geothermal_potential = st.selectbox("Geothermal Potential", ["Low", "Medium", "High"])
            biomass_availability = st.selectbox("Biomass Resource Availability", ["Low", "Medium", "High"])
        
        # Calculate renewable energy recommendations
        if st.button("Generate Recommendations") or st.session_state.renewable_recommendation is not None:
            with st.spinner("Analyzing renewable energy potential..."):
                if st.session_state.renewable_recommendation is None:
                    geo_features = {
                        'solar_irradiance': avg_solar_irradiance,
                        'wind_speed': avg_wind_speed,
                        'elevation_diff': elevation_diff,
                        'water_bodies': water_bodies,
                        'geothermal_potential': geothermal_potential,
                        'biomass_availability': biomass_availability
                    }
                    
                    st.session_state.renewable_recommendation = recommend_renewable_energy(
                        st.session_state.selected_area,
                        st.session_state.energy_consumption,
                        geo_features
                    )
            
            # Display results
            st.subheader("Renewable Energy Potential")
            
            # Energy potential visualization
            fig = go.Figure()
            
            for source, data in st.session_state.renewable_recommendation['sources'].items():
                fig.add_trace(go.Bar(
                    x=[source],
                    y=[data['score']],
                    name=source,
                    text=[f"{data['score']:.1f}/10"],
                    textposition='auto',
                    marker_color=data['color']
                ))
            
            fig.update_layout(
                title="Renewable Energy Potential Score (0-10)",
                xaxis_title="Energy Source",
                yaxis_title="Suitability Score",
                yaxis=dict(range=[0, 10]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed recommendations
            st.subheader("Recommended Energy Mix")
            
            col1, col2 = st.columns(2)
            
            with col1:
                mix_fig = go.Figure(data=[go.Pie(
                    labels=list(st.session_state.renewable_recommendation['energy_mix'].keys()),
                    values=list(st.session_state.renewable_recommendation['energy_mix'].values()),
                    hole=.3
                )])
                mix_fig.update_layout(title="Recommended Energy Mix")
                st.plotly_chart(mix_fig, use_container_width=True)
            
            with col2:
                st.markdown("### Key Findings")
                st.write(st.session_state.renewable_recommendation['summary'])
                
                st.markdown("### Implementation Notes")
                for note in st.session_state.renewable_recommendation['implementation_notes']:
                    st.write(f"- {note}")
            
            if st.button("View Final Report"):
                analysis_stage = "Report"
                st.rerun()

# Final Report Stage
elif analysis_stage == "Report":
    st.header("Comprehensive Development Report")
    
    if (st.session_state.housing_analysis is None or 
        st.session_state.population_estimate is None or 
        st.session_state.energy_consumption is None or 
        st.session_state.renewable_recommendation is None):
        
        st.warning("Please complete all analysis stages first.")
        if st.button("Return to Area Selection"):
            analysis_stage = "Area Selection"
            st.rerun()
    else:
        # Create tabs for different sections of the report
        tab1, tab2, tab3, tab4 = st.tabs([
            "Area Overview", 
            "Housing Development", 
            "Population & Energy", 
            "Renewable Energy"
        ])
        
        with tab1:
            st.subheader("Area Overview")
            st.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
            st.write(f"Area Size: {st.session_state.population_estimate['area_size']:.2f} km²")
            
            # Map visualization would go here in a complete implementation
            st.write("Map visualization of the selected area:")
            m = folium.Map(location=[
                np.mean([coord[0] for coord in st.session_state.selected_area]),
                np.mean([coord[1] for coord in st.session_state.selected_area])
            ], zoom_start=12)
            
            # Add polygon of selected area
            folium.Polygon(
                locations=st.session_state.selected_area,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.2
            ).add_to(m)
            
            folium_static(m, width=800, height=400)
        
        with tab2:
            st.subheader("Housing Development Suitability")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = create_suitability_plot(st.session_state.housing_analysis)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("Overall Suitability Assessment:")
                st.write(st.session_state.housing_analysis['recommendation'])
                st.metric("Overall Score", f"{st.session_state.housing_analysis['overall_score']:.2f}/10")
                
                st.write("Key Strengths:")
                strengths = sorted(
                    st.session_state.housing_analysis['factor_scores'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                for factor, score in strengths:
                    st.write(f"- {factor}: {score:.2f}/10")
                
                st.write("Areas for Consideration:")
                weaknesses = sorted(
                    st.session_state.housing_analysis['factor_scores'].items(), 
                    key=lambda x: x[1]
                )[:3]
                for factor, score in weaknesses:
                    st.write(f"- {factor}: {score:.2f}/10")
        
        with tab3:
            st.subheader("Population and Energy Consumption Projections")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Population Estimates:")
                st.metric("Total Population", f"{st.session_state.population_estimate['total_population']:,.0f}")
                st.metric("Housing Units", f"{st.session_state.population_estimate['housing_units']:,.0f}")
                st.metric("Population Density", f"{st.session_state.population_estimate['population_density']:.2f} people/km²")
                
                pop_fig = create_population_plot(st.session_state.population_estimate)
                st.plotly_chart(pop_fig, use_container_width=True)
            
            with col2:
                st.write("Energy Consumption Projections:")
                st.metric("Annual Consumption", f"{st.session_state.energy_consumption['total_consumption']:,.0f} kWh")
                st.metric("Peak Load", f"{st.session_state.energy_consumption['peak_load']:.2f} MW")
                st.metric("CO2 Emissions (Current Grid Mix)", f"{st.session_state.energy_consumption['co2_emissions']:,.0f} tonnes/year")
                
                energy_fig = create_energy_plot(st.session_state.energy_consumption)
                st.plotly_chart(energy_fig, use_container_width=True)
        
        with tab4:
            st.subheader("Renewable Energy Recommendations")
            
            # Energy mix visualization
            mix_fig = go.Figure(data=[go.Pie(
                labels=list(st.session_state.renewable_recommendation['energy_mix'].keys()),
                values=list(st.session_state.renewable_recommendation['energy_mix'].values()),
                hole=.3
            )])
            mix_fig.update_layout(title="Recommended Energy Mix")
            st.plotly_chart(mix_fig, use_container_width=True)
            
            st.write("Renewable Energy Assessment:")
            st.write(st.session_state.renewable_recommendation['summary'])
            
            # Display detailed recommendations for top sources
            st.subheader("Detailed Recommendations")
            
            top_sources = sorted(
                st.session_state.renewable_recommendation['sources'].items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )
            
            for source, data in top_sources:
                with st.expander(f"{source} - Score: {data['score']:.1f}/10"):
                    st.write(data['recommendation'])
                    st.write("Implementation considerations:")
                    for point in data['considerations']:
                        st.write(f"- {point}")
        
        # Download report option
        st.subheader("Download Complete Report")
        
        if st.button("Generate PDF Report"):
            st.info("In a complete implementation, this would generate a downloadable PDF with all analysis results.")
            
        # Reset analysis option
        if st.button("Start New Analysis"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            analysis_stage = "Area Selection"
            st.rerun()
