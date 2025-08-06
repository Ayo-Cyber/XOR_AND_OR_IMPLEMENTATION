import streamlit as st
import numpy as np
import os

# Set environment variables to prevent threading issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow not available. Only Numpy implementation will work.")

from logic_gates import train_numpy_model
if TENSORFLOW_AVAILABLE:
    from logic_gates import train_tensorflow_model

st.title("Logic Gate Neural Network")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'implementation' not in st.session_state:
    st.session_state.implementation = None
if 'gate' not in st.session_state:
    st.session_state.gate = None

# Sidebar controls
implementation_options = ["Numpy"]
if TENSORFLOW_AVAILABLE:
    implementation_options.append("TensorFlow")

implementation = st.sidebar.selectbox("Choose Implementation", implementation_options)
gate = st.sidebar.selectbox("Choose Logic Gate", ["AND", "OR", "XOR"])

# Train model button
if st.sidebar.button("Train Model"):
    try:
        st.session_state.model = None  # Reset model
        
        if implementation == "Numpy":
            with st.spinner("Training Numpy Model..."):
                model = train_numpy_model(gate)
                st.session_state.model = model
                st.session_state.implementation = "Numpy"
                st.session_state.gate = gate
                st.success("Numpy Model Trained!")
                
        elif implementation == "TensorFlow" and TENSORFLOW_AVAILABLE:
            with st.spinner("Training TensorFlow Model..."):
                # Clear any existing TensorFlow session
                tf.keras.backend.clear_session()
                model = train_tensorflow_model(gate)
                st.session_state.model = model
                st.session_state.implementation = "TensorFlow"
                st.session_state.gate = gate
                st.success("TensorFlow Model Trained!")
                
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        st.session_state.model = None

# Display results if model exists
if st.session_state.model is not None:
    st.subheader(f"Truth Table - {st.session_state.gate} Gate ({st.session_state.implementation})")
    
    # Test inputs
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    try:
        if st.session_state.implementation == "Numpy":
            predictions = st.session_state.model.predict(X)
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            predictions = np.round(predictions).astype(int)
        else:  # TensorFlow
            predictions = st.session_state.model.predict(X, verbose=0)
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            predictions = np.round(predictions).astype(int)

        # Display truth table
        st.write("**Input 1 | Input 2 | Prediction**")
        for i in range(len(X)):
            st.write(f"   {X[i][0]}    |    {X[i][1]}   |     {predictions[i]}")

        # Interactive testing
        st.subheader("Test the Model")
        col1, col2 = st.columns(2)
        
        with col1:
            input1 = st.selectbox("Input 1", [0, 1])
        with col2:
            input2 = st.selectbox("Input 2", [0, 1])

        if st.button("Predict"):
            try:
                input_data = np.array([[input1, input2]])
                
                if st.session_state.implementation == "Numpy":
                    prediction = st.session_state.model.predict(input_data)
                else:  # TensorFlow
                    prediction = st.session_state.model.predict(input_data, verbose=0)
                
                if prediction.ndim > 1:
                    prediction = prediction.flatten()
                prediction = np.round(prediction).astype(int)[0]
                
                st.write(f"**Prediction: {prediction}**")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        st.session_state.model = None
else:
    st.info("Please train a model using the sidebar controls.")