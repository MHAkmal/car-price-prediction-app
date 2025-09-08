import streamlit as st
import polars as pl
import joblib
from streamlit_option_menu import option_menu
import pandas as pd # Using pandas for easier display formatting in st.dataframe
import streamlit.components.v1 as components # Added for HTML embedding

# --- Page layout configuration (called only once) ---
st.set_page_config(
    page_title="Car Price Predictor",
    layout="wide"
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 300px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


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
        menu_title="Main Menu",  # required
        options=["Home", "Prediction", "Model Notebook"],  # required
        icons=["house", "clipboard-data", "book"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )

# --- Home Page ---
if selection == "Home":
    st.title("🚗 Car Price Prediction")
    st.markdown("""
    Welcome to the Car Price Prediction app! This application predicts the estimated price of a car based on its various attributes.
    """)
    
    st.warning("Navigate to the **Prediction** tab from the sidebar to try the prediction yourself!", icon="👈")
    
    st.divider()

    # --- Business Impact Section ---
    st.header("Business Impact")
    st.subheader("Optimizing Pricing in the Automobile Industry")
    st.markdown("Predicting car prices based on specifications can help automobile companies optimize their pricing strategies, leading to increased sales and revenue.")

    st.subheader("Model Performance")
    st.info("The prediction model is a **Linear Regression** algorithm.")

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
if selection == "Model Notebook":
    st.title("🔬 Car Price Prediction Model Notebook")
    st.write("This is a live display of the Jupyter Notebook used for data exploration, cleaning, and model building, converted to HTML.")

    notebook_filename = "car-price-prediction-ml-model_pandas.html"
    try:
        # Open the HTML file and read its content
        with open(notebook_filename, "r", encoding="utf-8") as f:
            html_data = f.read()

        # Display the HTML in an iframe
        st.header("Notebook Output")
        components.html(html_data, width=None, height=800, scrolling=True)

    except FileNotFoundError:
        st.error(f"File not found: '{notebook_filename}'. Please ensure the notebook has been exported to HTML and is in the same directory as this script.")

    st.info("The notebook above is a static HTML file. For interactive elements, consider porting the code directly into this Streamlit app.")
