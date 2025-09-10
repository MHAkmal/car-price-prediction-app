import streamlit as st
import polars as pl
import joblib
from streamlit_option_menu import option_menu
import pandas as pd 
import streamlit.components.v1 as components 
import base64 

# --- Page layout configuration --
st.set_page_config(
    page_title="Car Price Predictor",
    layout="wide"
)

# --- Custom CSS for styling ---
st.markdown(
    """
    <style>
    /* Sidebar width */
    [data-testid="stSidebar"] {
        width: 300px !important;
    }
    
    /* Container to help with alignment */
    .connect-box {
        display: flex;
        flex-direction: column;
        align-items: center; /* Center alignment */
        justify-content: center;
        height: 100%;
    }

    /* Style for connect links */
    .connect-container {
        display: flex;
        flex-direction: column;
        gap: 10px; /* Space between links */
    }
    .connect-link {
        display: flex;
        align-items: center;
        gap: 8px; /* Space between icon and text */
        text-decoration: none;
        font-weight: bold;
        color: #FFFFFF !important; /* White text for links */
    }
    .connect-link:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SVGs for Icons (Base64 Encoded) ---
def get_svg_as_b64(svg_raw):
    return base64.b64encode(svg_raw.encode('utf-8')).decode()

linkedin_svg = get_svg_as_b64('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#0077B5" stroke="currentColor" stroke-width="0" stroke-linecap="round" stroke-linejoin="round"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect x="2" y="9" width="4" height="12"></rect><circle cx="4" cy="4" r="2"></circle></svg>')
github_svg = get_svg_as_b64('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#FFFFFF" stroke="currentColor" stroke-width="0" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>')
x_svg = get_svg_as_b64('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 16 16" fill="#FFFFFF"><path d="M12.6.75h2.454l-5.36 6.142L16 15.25h-4.937l-3.867-5.07-4.425 5.07H.316l5.733-6.57L0 .75h5.063l3.495 4.633L12.602.75Zm-1.283 12.95h1.46l-7.48-10.74h-1.55l7.57 10.74Z"/></svg>')
threads_svg = get_svg_as_b64('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22a10 10 0 1 1 0-20 10 10 0 0 1 0 20Z"></path><path d="M16.5 8.5c-.7-1-1.8-1.5-3-1.5s-2.3.5-3 1.5"></path><path d="M16.5 15.5c-.7 1-1.8 1.5-3 1.5s-2.3-.5-3-1.5"></path><path d="M8.5 12a5.5 5.5 0 1 0 7 0 5.5 5.5 0 0 0-7 0Z"></path></svg>')

# --- Caching the Model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained car price prediction model pipeline."""
    try:
        model = joblib.load('car_price_prediction_model.pkl')
        return model
    except FileNotFoundError:
        st.error("The model file 'car_price_prediction_model.pkl' was not found. Please make sure the model is in the same directory as the script.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Load the model
model = load_model()

# --- Initialize session state for prediction history ---
if 'prediction_log' not in st.session_state:
    st.session_state.prediction_log = []


# --- Sidebar Navigation ---
with st.sidebar:
    selection = option_menu(
        menu_title="Main Menu",
        options=["Home", "Prediction", "Notebook"],
        icons=["house", "clipboard-data", "book"],
        menu_icon="cast",
        default_index=0,
    )

# --- Home Page ---
if selection == "Home":

    # 1. Top Section with 75/25 Split ---
    top_col1, top_col2 = st.columns([0.75, 0.25])

    with top_col1:
        st.title("🚗 Car Price Prediction")
        st.markdown("""
        Welcome to the Car Price Prediction app! This application predicts the estimated price of a car based on its various attributes.
        """)
        st.warning("Navigate to the **Prediction** tab from the sidebar to try the prediction yourself!", icon="👈")

    with top_col2:
        st.markdown('<div class="connect-box">', unsafe_allow_html=True)
        st.subheader("🔗 Connect With Me")
        linkedin_link = f'<a href="https://www.linkedin.com/in/mhakmal/" class="connect-link"><img src="data:image/svg+xml;base64,{linkedin_svg}" width="24"><span>MHAkmal</span></a>'
        github_link = f'<a href="https://github.com/MHAkmal" class="connect-link"><img src="data:image/svg+xml;base64,{github_svg}" width="24"><span>MHAkmal</span></a>'
        x_link = f'<a href="https://x.com/akmal621" class="connect-link"><img src="data:image/svg+xml;base64,{x_svg}" width="24"><span>MHAkmal</span></a>'
        threads_link = f'<a href="https://www.threads.com/@akmal621?__pwa=1" class="connect-link"><img src="data:image/svg+xml;base64,{threads_svg}" width="24"><span>MHAkmal</span></a>'
        st.markdown(f'<div class="connect-container">{linkedin_link}{github_link}{x_link}{threads_link}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


    st.divider()

    # --- Business Problem and Objective ---
    bp_col1, obj_col2 = st.columns(2)
    with bp_col1:
        st.header("Business Problem")
        st.write("How to increase revenue from car sales by predicting accurate car price based on specifications?")
    
    with obj_col2:
        st.header("Objective")
        st.write("Build a regression machine learning model to predict car price.")
    
    st.divider()

    # --- Business Impact Section ---
    st.header("Business Impact")
    st.subheader("Model Performance (Model: Linear Regression)")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(label="Mean Absolute Error (MAE)", value="$1,970")
    with m_col2:
        st.metric(label="Root Mean Squared Error (RMSE)", value="$2,822")

    st.subheader("Sales Impact Scenario (per 100 cars)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Before Modeling")
        st.markdown("""
        - The company struggles with accurate car price prediction.
        - Pricing may not align with market expectations.
        """)
        st.metric(label="Cars Sold (out of 100)", value="50")

    with col2:
        st.markdown("#### After Modeling")
        st.markdown("""
        - The company can set prices that are closer to the market value.
        - Assuming the accurate car price are those with error/diff
        - With a **93% accuracy rate**, pricing becomes highly competitive.
        """)
        st.metric(label="Cars Sold (out of 100)", value="93", delta="43 cars")

    st.success("Implementing this model could lead to an estimated **43 additional cars sold** for every 100 cars in inventory.")


# --- Prediction Page ---
if selection == "Prediction" and model is not None:
    st.title("Predict Car Price")
    st.warning("Adjust the parameters in the sections below, then scroll down and click the 'Predict Price' button to see the result.", icon="ℹ️")
    st.header("Enter Car Details")

    # Create columns for a more organized layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Categorical Features")
        fueltype = st.selectbox('Fuel Type', ['gas', 'diesel'], key='fueltype')
        aspiration = st.selectbox('Aspiration', ['std', 'turbo'], key='aspiration')
        doornumber = st.selectbox('Number of Doors', ['two', 'four'], key='doornumber')
        carbody = st.selectbox('Car Body', ['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop'], key='carbody')
        drivewheel = st.selectbox('Drive Wheel', ['rwd', 'fwd', '4wd'], key='drivewheel')
        enginelocation = st.selectbox('Engine Location', ['front', 'rear'], key='enginelocation')
        enginetype = st.selectbox('Engine Type', ['dohc', 'ohcv', 'ohc', 'l', 'rotor', 'ohcf', 'dohcv'], key='enginetype')
        cylindernumber = st.selectbox('Number of Cylinders', ['four', 'six', 'five', 'three', 'twelve', 'two', 'eight'], key='cylindernumber')
        fuelsystem = st.selectbox('Fuel System', ['mpfi', '2bbl', 'mfi', '1bbl', 'spfi', '4bbl', 'idi', 'spdi'], key='fuelsystem')

    with col2:
        st.subheader("Numerical Features")
        symboling = st.slider('Symboling', min_value=-2, max_value=3, value=0, key='symboling')
        wheelbase = st.slider('Wheelbase (in)', 86.6, 120.9, 98.8, key='wheelbase')
        carlength = st.slider('Car Length (in)', 141.1, 208.1, 174.0, key='carlength')
        carwidth = st.slider('Car Width (in)', 60.3, 72.3, 65.9, key='carwidth')
        carheight = st.slider('Car Height (in)', 47.8, 59.8, 53.7, key='carheight')
        curbweight = st.slider('Curb Weight (lbs)', 1488, 4066, 2555, key='curbweight')
        enginesize = st.slider('Engine Size (ci)', 61, 326, 127, key='enginesize')
        boreratio = st.slider('Bore Ratio', 2.54, 3.94, 3.33, key='boreratio')
        stroke = st.slider('Stroke', 2.07, 4.17, 3.26, key='stroke')
        compressionratio = st.slider('Compression Ratio', 7.0, 23.0, 10.0, key='compressionratio')
        horsepower = st.slider('Horsepower', 48, 288, 104, key='horsepower')
        peakrpm = st.slider('Peak RPM', 4150, 6600, 5125, key='peakrpm')
        citympg = st.slider('City MPG', 13, 49, 25, key='citympg')
        highwaympg = st.slider('Highway MPG', 16, 54, 31, key='highwaympg')

        st.divider()
        calculated_car_area = carlength * carwidth
        calculated_car_volume = calculated_car_area * carheight
        st.metric(label="Calculated Car Area", value=f"{calculated_car_area:,.2f}")
        st.metric(label="Calculated Car Volume", value=f"{calculated_car_volume:,.2f}")


    # --- Prediction Logic ---
    if st.button('Predict Price', type="primary"):
        current_input = {
            'fueltype': fueltype, 'aspiration': aspiration, 'doornumber': doornumber, 'carbody': carbody, 'drivewheel': drivewheel,
            'enginelocation': enginelocation, 'enginetype': enginetype, 'cylindernumber': cylindernumber, 'fuelsystem': fuelsystem,
            'symboling': symboling, 'wheelbase': wheelbase, 'carlength': carlength, 'carwidth': carwidth, 'carheight': carheight,
            'curbweight': curbweight, 'enginesize': enginesize, 'boreratio': boreratio, 'stroke': stroke, 'compressionratio': compressionratio,
            'horsepower': horsepower, 'peakrpm': peakrpm, 'citympg': citympg, 'highwaympg': highwaympg,
            'car_area': calculated_car_area, 'car_volume': calculated_car_volume
        }
        is_duplicate = any(log_entry['input'] == current_input for log_entry in st.session_state.prediction_log)
        if is_duplicate:
            st.warning("This exact prediction has already been made. Please see the history below.")
        else:
            input_data = pl.DataFrame({key: [value] for key, value in current_input.items()})
            prediction = model.predict(input_data)[0]
            st.subheader("Predicted Car Price")
            st.success(f"**${prediction:,.2f}**")
            st.session_state.prediction_log.insert(0, {'input': current_input, 'prediction': prediction})

    # --- Display Prediction History ---
    st.divider()
    st.header("Prediction History")
    if st.session_state.prediction_log:
        history_df = pd.DataFrame([entry['input'] | {'Predicted Price': entry['prediction']} for entry in st.session_state.prediction_log])
        st.dataframe(history_df.style.format({'Predicted Price': '${:,.2f}', 'car_area': '{:,.2f}', 'car_volume': '{:,.2f}'}), use_container_width=True)
    else:
        st.info("No predictions have been made in this session yet.")
elif selection == "Prediction" and model is None:
    st.warning("The prediction model is not available. Please check for error messages when the app started.")

# --- Notebook Page ---
if selection == "Notebook":
    # 2. Changed notebook title
    st.title("Car Price Prediction Model Notebook")
    # 3. Changed notebook description
    st.write("This is a live display of the Marimo Notebook used for data exploration, cleaning, and model building, converted to HTML.")
    
    # 4. Changed info box text
    st.info("The notebook below is a static HTML file.")
    
    notebook_filename = "car-price-prediction-ml-model_pandas.html"
    try:
        with open(notebook_filename, "r", encoding="utf-8") as f:
            html_data = f.read()
        components.html(html_data, width=None, height=800, scrolling=True)
    except FileNotFoundError:
        st.error(f"File not found: '{notebook_filename}'. Please ensure the notebook has been exported to HTML and is in the same directory as this script.")

