import streamlit as st
import numpy as np
import pickle
import time

# Page configuration
st.set_page_config(
    page_title="Academic Class Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling and interactive animations
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Container Styling */
    .header-container {
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .header-container h1 {
        margin: 0;
        font-size: 2.25rem;
        font-weight: 700;
    }
    .header-container p {
        margin-top: 0.5rem;
        opacity: 0.9;
        font-size: 1rem;
    }

    /* Input Card Styling */
    div[data-testid="stForm"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 2rem;
        background-color: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    /* Button Styling with Animation */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(79, 70, 229, 0.3);
        background: linear-gradient(135deg, #4338CA 0%, #6D28D9 100%);
        color: #ffffff;
    }

    /* Metric Display Box */
    .result-card {
        padding: 1.5rem;
        background-color: #ffffff;
        border-radius: 10px;
        border-left: 5px solid #4F46E5;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-top: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Function to load the saved model pickle file
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# App Header
st.markdown("""
    <div class="header-container">
        <h1>🎓 Academic Performance Classifier</h1>
        <p>Predict classification targets using your trained K-Nearest Neighbors model.</p>
    </div>
""", unsafe_allow_html=True)

# Main layout logic
try:
    model = load_model()
    
    # Sidebar Info
    with st.sidebar:
        st.header("⚙️ Model Details")
        st.write("**Algorithm:** K-Nearest Neighbors")
        st.write("**Neighbors (K):**", model.n_neighbors)
        st.write("**Metric:**", model.metric)
        st.divider()
        st.info("💡 Adjust the slider inputs to predict the output class based on subject performance.")

    # Main Input Form
    with st.form("prediction_form"):
        st.subheader("📝 Enter Subject Marks")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hindi = st.number_input("Hindi", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
            english = st.number_input("English", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
            science = st.number_input("Science", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
            maths = st.number_input("Maths", min_value=0.0, max_value=100.0, value=90.0, step=1.0)

        with col2:
            history = st.number_input("History", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            geography = st.number_input("Geography", min_value=0.0, max_value=100.0, value=78.0, step=1.0)
            
            # Automatically calculate total, but allow custom entry if needed
            calculated_total = float(hindi + english + science + maths + history + geography)
            total = st.number_input("Total", min_value=0.0, max_value=600.0, value=calculated_total, step=1.0)

        submit_btn = st.form_submit_button("🚀 Run Prediction")

    # Prediction Action & Effect Handling
    if submit_btn:
        # Visual loader animation delay
        with st.spinner("Processing features through KNN pipeline..."):
            time.sleep(0.6)  # Short artificial delay for smooth effect
        
        # Prepare feature vector matching training array shape
        features = np.array([[hindi, english, science, maths, history, geography, total]])
        
        # Inference
        prediction = model.predict(features)[0]
        
        # Celebration animation effect
        st.balloons()
        
        # Output card display
        st.markdown(f"""
            <div class="result-card">
                <h3 style="margin:0; color:#1F2937;">Prediction Outcome</h3>
                <p style="font-size:1.8rem; font-weight:700; color:#4F46E5; margin-top:0.5rem; margin-bottom:0;">
                    Predicted Target: Class {prediction}
                </p>
            </div>
        """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("Error: `model.pkl` file not found. Ensure `model.pkl` is located in the root directory.")
except Exception as e:
    st.error(f"An error occurred while running the application: {e}")
