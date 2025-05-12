import streamlit as st
import matplotlib.pyplot as plt
from model import train_perceptron
from utils import generate_hooke_data, plot_results

st.set_page_config(page_title="Perceptrón y Ley de Hooke", layout="centered")

st.title("📉 Perceptrón que aprende la Ley de Hooke")

st.markdown("""
Este dashboard muestra cómo un perceptrón simple puede aprender la ley de Hooke:  
\\( F = -k \\cdot x \\)  
Usamos datos sintéticos generados con una constante de resorte \\( k \\).
""")

# Parámetros
k_real = st.slider("Constante del resorte (k)", min_value=0.1, max_value=10.0, value=3.5, step=0.1)
lr = st.number_input("Tasa de aprendizaje", min_value=0.001, max_value=1.0, value=0.01)
epochs = st.slider("Épocas de entrenamiento", min_value=100, max_value=2000, value=500, step=100)

# Generar datos
x, F = generate_hooke_data(k_real)

# Entrenar modelo
model, loss_hist = train_perceptron(x, F, lr, epochs)

# Predicciones
with torch.no_grad():
    pred = model(x)
    learned_k = -model.weight.item()

# Mostrar resultados
st.markdown(f"**Constante aprendida por el perceptrón:** k = {learned_k:.4f}")
plot_results(x, F, pred, k_real, learned_k)
