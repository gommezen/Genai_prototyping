# import packages
import streamlit as st
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt


# # Helper function to get dataset path
# def get_dataset_path():
#     # Get the current script directory
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     # Construct the path to the CSV file
#     csv_path = os.path.join(current_dir, "data", "customer_reviews_long_clean.csv")
#     return csv_path

def get_dataset_path(filename="customer_reviews_long_clean.csv"):
    """
    Resolve the absolute path to a dataset stored in the project's /data folder.
    Works even when this script is located inside an /apps directory.
    """

    # Directory of the current script (e.g., .../genai-prototype/apps/)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Project root = parent folder of /apps
    project_root = os.path.dirname(script_dir)

    # Final dataset path: .../genai-prototype/data/<filename>
    data_path = os.path.join(project_root, "data", filename)

    # Validate that the dataset actually exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at: {data_path}\n"
            f"Expected location:\n"
            f"genai-prototype/data/{filename}"
        )

    return data_path



# Helper function to clean text
# This function takes a text string as input, converts it to lowercase, removes leading and trailing whitespace,
# and removes any characters that are not alphanumeric or whitespace.
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text


st.title("Hello, GenAI!")
st.write("This is your GenAI-powered data processing app.")

# Layout two buttons side by side
col1, col2 = st.columns(2)

# Ingest Dataset button
with col1:
    if st.button("📥 Ingest Dataset"):
        try:
            csv_path = get_dataset_path()
            # explain what this line does
            st.session_state["df"] = pd.read_csv(csv_path) # Load the CSV file into a DataFrame and store it in session state
            st.success("Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")

# Parse Reviews button
with col2:
    if st.button("🧹 Parse Reviews"):
        if "df" in st.session_state:
            st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
            st.success("Reviews parsed and cleaned!")
        else:
            st.warning("Please ingest the dataset first.")

# Display the dataset if it exists
if "df" in st.session_state:
    
    # Product filter dropdown
    st.subheader("🔍 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reviews for {product}")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    st.dataframe(filtered_df)

    
    st.subheader("Sentiment Score by Product")
    grouped = st.session_state["df"].groupby(["PRODUCT"])["SENTIMENT_SCORE"].mean()
    st.bar_chart(grouped)
    
    st.subheader(f"Sentiment Score Distribution for {product}")
    
    # Create matplotlib histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(filtered_df["SENTIMENT_SCORE"], bins=10, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Sentiment Scores')
    st.pyplot(fig)
    
    # Create Plotly histogram
    fig = px.histogram(
        filtered_df, 
        x="SENTIMENT_SCORE", 
        nbins=10,
        title="Distribution of Sentiment Scores",
        labels={"SENTIMENT_SCORE": "Sentiment Score", "count": "Frequency"}
    )
    fig.update_layout(
        xaxis_title="Sentiment Score",
        yaxis_title="Frequency",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

