# Logic Gate Neural Network

This project provides a web-based application to train and test neural network models for basic logic gates (AND, OR, XOR). You can choose between two implementations: one built from scratch using NumPy, and another using the popular TensorFlow library.

## Features

- **Interactive Web Interface:** A user-friendly interface built with Streamlit to visualize the training and testing of the models.
- **Multiple Logic Gates:** Train models for AND, OR, and XOR logic gates.
- **Dual Implementations:** Choose between a NumPy-based neural network and a TensorFlow-based one.
- **Real-time Prediction:** Test the trained models with custom inputs and see the predictions instantly.
- **Dynamic Truth Tables:** View the complete truth table for the selected logic gate as predicted by the trained model.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ayo-Cyber/XOR_AND_OR_IMPLEMENTATION.git
   cd XOR_AND_OR_IMPLEMENTATION
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Once the installation is complete, you can run the Streamlit application with the following command:

```bash
streamlit run app.py
```

This will open the application in your web browser. From there, you can use the sidebar to:

1. **Choose an implementation** (NumPy or TensorFlow).
2. **Select a logic gate** (AND, OR, or XOR).
3. **Train the model** by clicking the "Train Model" button.

After the model is trained, the application will display the truth table and allow you to make real-time predictions.

## Implementations

### NumPy

The NumPy-based implementation is a simple neural network built from scratch. It uses basic matrix operations to perform the forward and backward passes. This implementation is great for understanding the core concepts of a neural network.

- **`src/neural_network.py`**: Contains the `NeuralNetwork` class that handles the model's architecture, training, and prediction.
- **`src/logic_gates.py`**: Contains the `train_numpy_model` function that prepares the data and trains the NumPy model for the selected logic gate.

### TensorFlow

The TensorFlow-based implementation uses the Keras API to build and train the neural network. This implementation is more powerful and efficient, especially for more complex models.

- **`src/tensorflow_model.py`**: Contains the functions to create and train the TensorFlow models for the different logic gates.
- **`src/logic_gates.py`**: Contains the `train_tensorflow_model` function that prepares the data and trains the TensorFlow model.

## Dependencies

The project's dependencies are listed in the `requirements.txt` file:

- **pandas**
- **numpy**
- **tensorflow**
- **streamlit**