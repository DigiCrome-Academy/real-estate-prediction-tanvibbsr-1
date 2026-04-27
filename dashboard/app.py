"""
Real Estate Price Prediction Dashboard

Run with: streamlit run dashboard/app.py

This dashboard provides:
- Price prediction using trained models
- Property similarity recommendations
- Market segmentation visualization
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(
    page_title="Real Estate Price Prediction Engine",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Real Estate Price Prediction Engine")
st.markdown("---")

# =============================================================================
# Sidebar - Navigation
# =============================================================================
page = st.sidebar.selectbox(
    "Navigate",
    ["Price Prediction", "Property Recommendations", "Market Segmentation"]
)

# =============================================================================
# TODO: Load your trained models and data
# =============================================================================
# Hints:
#   from src.data_loader import load_housing_data, preprocess_features
#   from src.ensemble import load_model
#   model = load_model('models/best_model.joblib')
from src.recommendation import knn_recommend, content_based_recommend, compute_property_similarity
from src.data_loader import load_housing_data, preprocess_features
from src.ensemble import load_model
# Load data and preprocess features
data = load_housing_data()
features_scaled, y, feature_names, scaler = preprocess_features(data)
# Load trained model
model = load_model('models/best_model.joblib')


if page == "Price Prediction":
    st.header("💰 Price Prediction")
    st.write("Enter property features to get a price estimate.")

    # TODO: Create input widgets for each feature
    # Example:
    # med_income = st.slider("Median Income (area)", 0.0, 15.0, 3.0)
    # house_age = st.slider("House Age", 1, 52, 20)
    # ...
    widgets = {}
    feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    for feature in feature_names:
        if feature in ['MedInc']:
            widgets[feature] = st.slider(feature, 0.0, 15.0, 3.0)
        elif feature in ['HouseAge']:
            widgets[feature] = st.slider(feature, 1, 52, 20)
        elif feature in ['AveRooms', 'AveBedrms', 'AveOccup']:
            widgets[feature] = st.slider(feature, 0.0, 10.0, 2.0)
        elif feature in ['Population']:
            widgets[feature] = st.slider(feature, 0, 50000, 1000)
        elif feature in ['Latitude']:
            widgets[feature] = st.slider(feature, 32.0, 42.0, 37.0)
        elif feature in ['Longitude']:
            widgets[feature] = st.slider(feature, -125.0, -114.0, -120.0)

    # TODO: When user clicks "Predict", run the model
    # if st.button("Predict Price"):
    #     features = np.array([[med_income, house_age, ...]])
    #     features_scaled = scaler.transform(features)
    #     prediction = model.predict(features_scaled)
    #     st.success(f"Estimated Price: ${prediction[0] * 100000:,.0f}")
    if st.button("Predict Price"):
        features = np.array([[widgets[feature] for feature in feature_names]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        st.success(f"Estimated Price: ${prediction[0] * 100000:,.0f}")

    st.info("⚠️ Implement the prediction logic in src/ensemble.py first, "
            "then load your trained model here.")


elif page == "Property Recommendations":
    st.header("🔍 Property Recommendations")
    st.write("Find similar properties based on features.")

    # TODO: Let user select a property index or input features
    # TODO: Show top-N similar properties using your recommendation system
    property_index = st.number_input("Enter Property Index (0-20639)", min_value=0, max_value=20639, value=0)   
    if st.button("Show Recommendations"):  
        property_features = features_scaled[property_index].reshape(1, -1)
        similar_properties = knn_recommend(property_features, features_scaled, top_n=5)
        st.write("Top 5 Similar Properties:")
        st.dataframe(similar_properties)

    st.info("⚠️ Implement the recommendation logic in src/recommendation.py first.")


elif page == "Market Segmentation":
    st.header("📊 Market Segmentation")
    st.write("Explore property market segments identified by clustering.")

    # TODO: Load clustering results
    # TODO: Show PCA 2D scatter plot with cluster colors
    # TODO: Show cluster statistics table
    clustering_results = pd.read_csv('models/clustering_results.csv')  # Example path
    #Load clustering results
    st.subheader("PCA Scatter Plot")
    st.write("Visualize clusters in 2D PCA space.")

    # Show PCA 2D scatter plot with cluster colors
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=clustering_results['PCA1'], 
        y=clustering_results['PCA2'], 
        hue=clustering_results['Cluster'], 
        palette='Set2'
    )
    plt.title("PCA Scatter Plot of Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title='Cluster')
    st.pyplot(plt)
    st.subheader("Cluster Statistics")
    st.write("Summary statistics for each cluster.")
    cluster_stats = clustering_results.groupby('Cluster').mean()
    st.dataframe(cluster_stats)

    #Show cluster statistics table
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cluster_stats.index, y=cluster_stats['MedInc'], palette='Set2')
    plt.title("Average Median Income by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Average Median Income")
    st.pyplot(plt)

    st.info("⚠️ Implement the clustering logic in src/clustering.py first.")


# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown(
    "Built as part of the Real Estate Price Prediction Engine project. "
    "Uses the California Housing dataset from scikit-learn."
)
