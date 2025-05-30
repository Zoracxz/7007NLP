import streamlit as st
from src.recommend import get_recommendations, get_all_conditions

st.title("💊 Medicine Recommendation System")

# Get the list of all disease conditions for the dropdown
condition_list = get_all_conditions()

# Create a dropdown to select disease condition
selected_condition = st.selectbox("Select a disease condition:", condition_list)

# When the "Search" button is clicked
if st.button("Search"):
    result = get_recommendations(selected_condition)
    st.subheader("Top 10 Recommended Medicines:")
    for i, drug in enumerate(result, 1):
        st.write(f"{i}. {drug}")
