# Disease Prediction Project

## Summary
The Disease Prediction Project is designed to help healthcare providers, especially in developing regions, predict diseases based on selected symptoms. This project consists of two key files:
- **disease_prediction.py**: Implements the actual prediction app, allowing users to select symptoms and receive disease predictions.
- **main.ipynb**: Responsible for training the machine learning model used in the app, performing exploratory data analysis (EDA), and experimenting with different models before selecting the most suitable one for deployment.

Artificial Intelligence (AI) and Machine Learning (ML) are central to the project. The app uses a trained neural network to analyze symptoms provided by users and predict possible diseases, giving healthcare providers a fast and efficient way to support their patients.

## File Breakdown

### `disease_prediction.py`
- **Purpose**: This file runs the disease prediction web application. It takes user input in the form of symptoms and uses the trained machine learning model to predict the most likely disease.
- **Role of AI/ML**: The file loads a pre-trained ML model (from `TensorFlow/Keras`) that has learned to recognize patterns in symptom data and predict disease outcomes.

#### Libraries Used:
1. **Streamlit**:
   - Provides the interface for the app to be accessible via a web browser.
   - Manages the layout and design of the user interface.
   
2. **Pandas**:
   - Loads and manipulates the symptom data from CSV files.
   - Prepares input data for prediction by converting it into the required format.

3. **TensorFlow/Keras**:
   - Loads the pre-trained model used for making predictions.
   - Handles deep learning models and neural network structures for prediction tasks.

4. **Plotly**:
   - Provides visualization capabilities (optional) to present interactive visual data, although its use is not critical to the core prediction functionality.

5. **NumPy**:
   - Efficiently handles arrays and matrices of input data.

#### Methods Used:
1. **`display_dropdowns()`**:
   - Presents dropdown menus to users for symptom selection.
   - Dynamically updates the user interface to capture multiple symptoms.

2. **`on_select_change()`**:
   - Captures changes in user input (symptoms selected).
   - Triggers re-calculation or prediction when the user selects or changes symptoms.

### `main.ipynb`
- **Purpose**: This notebook is responsible for training the machine learning models, performing exploratory data analysis, and experimenting with different architectures to identify the best-performing model. The final selected model is then exported and used in the prediction app.
- **Role of AI/ML**: The notebook explores multiple machine learning models, comparing their performance on disease prediction tasks. The final selected model is the one that achieves the best balance between accuracy and computational efficiency.

#### Libraries Used:
1. **Pandas**:
   - Loads, cleans, and preprocesses the dataset.
   - Extracts relevant symptom and disease information for model training.

2. **Seaborn & Matplotlib**:
   - Used for data visualization during the EDA process.
   - Helps to identify correlations and patterns in the dataset.

3. **Scikit-learn**:
   - Provides tools for data preprocessing, including One-Hot Encoding and standardization of data.
   - Facilitates splitting the dataset into training and test sets for model evaluation.

4. **TensorFlow/Keras**:
   - Defines the neural network architectures (MLP, CNN, LSTM) used to predict diseases.
   - Trains and evaluates the performance of each model.

5. **Plotly**:
   - Visualizes model performance and data distribution.
   - Interactive plots help analyze the training results.

6. **NumPy**:
   - Efficiently manages large datasets and mathematical operations.
   
#### Models Explored:
1. **Multi-Layer Perceptron (MLP)**:
   - This model was implemented using the `Sequential` API in Keras. It was simple, using fully connected layers but achieved relatively lower accuracy compared to more complex models.
   - **Accuracy**: ~72% on the test set.

2. **Convolutional Neural Network (CNN)**:
   - The CNN model was also implemented using the `Sequential` API. It performed better than the MLP model due to its ability to capture local patterns in the data.
   - **Accuracy**: ~85%. This model showed improved accuracy, especially on complex diseases that involve multiple symptoms.

3. **Long Short-Term Memory (LSTM)**:
   - LSTMs were explored due to their ability to remember previous symptom patterns. However, this model was more computationally expensive and did not significantly outperform the CNN model.
   - **Accuracy**: ~83%. This model was not chosen due to its high computational cost and similar performance to CNNs.

4. **Selected Model - CNN**:
   - The CNN model was selected as the final model due to its high accuracy and efficient handling of symptom data. It performed well on the dataset, providing accurate predictions while maintaining computational efficiency for deployment on low-resource environments.

## Conclusion
The Disease Prediction Project is designed with resource-limited communities in mind, such as rural healthcare centers or developing countries where access to medical professionals may be scarce. Unlike WebMD or IBM Watson, which often have vague and broad symptom checkers requiring significant computational resources, this project allows users to select specific symptoms and receive accurate, quick predictions. Additionally, the CNN model selected in this project is computationally efficient, allowing it to run on devices with limited processing power, making it a practical solution for under-resourced areas.


