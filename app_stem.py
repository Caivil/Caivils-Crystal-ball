# GUI for linear model prediction
# importing packages
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
#%%

# st.logo
with st.container():
    left, mid, right = st.columns([3, 4, 3])  # Wider center for better balance
    with mid:
        st.image("website.png", width=400)
st.title("Caivil’s Data Crystal Ball")
st.write('Your Linear Regression Predictor')
st.write('By Caivil Ndobela')
st.write("   ")
st.write("   ")
st.write("   ")

        
# User detail
x = st.text_input("Please type your preferred name below:")
if x:  
    st.write(f"Hi {x}! Welcome to your personal linear predictor! Before we get started, please make sure your data is cleaned and ready to go. This will help the system work smoothly and give you the best results.")

# Add a section for adding data
st.header("Upload data as CSV")
data = st.file_uploader(" ", type="csv")

if data is not None:
    try:
        data = pd.read_csv(data)
        st.write("Data preview:")
        st.dataframe(data.head())
        
        if len(data.columns) >= 2:  # Check if there are at least 2 columns
            # Let user select columns
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X (independent) variable", data.columns)
            with col2:
                y_col = st.selectbox("Select Y (dependent) variable", data.columns)
            
            X = data[[x_col]]
            y = data[[y_col]]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Create and train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            coef = model.coef_[0][0]
            inter = model.intercept_[0]

            # Display model information
            st.header("Model Results")
            st.write(f"Coefficient (Slope): {coef:.4f}")  
            st.write(f"Y-Intercept: {inter:.4f}")     
            st.write(f"Regression Equation: y = {coef:.2f}x + {inter:.2f}")
            st.write(f"Mean Squared Error: {mse:.4f}")
            st.write(f"R-squared (the strength of correlation): {r2:.4f}")

            # Plot the results
            st.header(f"{x_col} vs {y_col}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X_test, y_test, color='black', label='Data points')
            ax.plot(X_test, y_pred, color='blue', linewidth=3, label='Trend')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{x_col} vs {y_col}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
        else:
            st.error("The uploaded file needs at least 2 columns to perform regression.")
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Add a contact section
st.header("Contact Information")
name = "Caivil Ndobela"
email = "Biocaivil@gmail.com"
st.write(f"You can reach {name} at {email}.")
