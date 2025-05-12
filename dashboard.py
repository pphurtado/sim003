import streamlit as st
import matplotlib.pyplot as plt
import torch
from model import train_perceptron
from utils import generate_hooke_data, plot_results

# Configuración de la página
st.set_page_config(page_title="Perceptrón y Ley de Hooke", layout="wide")

# Menú lateral (tipo hamburguesa)
with st.sidebar:
    st.title("Menú")
    seleccion = st.radio("Ir a:", ["Inicio", "Configuración","Visualización"])
    st.markdown("---")
    st.write("Opciones adicionales")
    
#Para Cargar Loto
import base64
# Cargar la imagen y convertirla en base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
img_base64 = get_base64_image("DeltaS_Logo_20250510_01.png")
# Insertar la imagen y el título
st.markdown(
    f"""
    <div style='display: flex; align-items: center;'>
        <img src='data:image/png;base64,{img_base64}' width='40' style='margin-right:10px;'/>
        <h1 style='display:inline;'>Dashboard for Statistical Physics</h1>
    </div>
    """,
    unsafe_allow_html=True
)
if seleccion == "Inicio":
    st.subheader("Bienvenido al Dashboard")
    st.write("Selecciona una opción del menú para comenzar.")

elif seleccion == "Configuración":
    st.subheader("⚙️ Configuración")
    st.write("Ajusta los parámetros según tus necesidades.")
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
    
    if opcion_extra:
        st.success("Modo avanzado activado.")

elif seleccion == "Visualización":
    st.subheader("📊 Visualización de Datos")
    st.write("Aquí podrías insertar un gráfico, tabla o resultado.")
    # Mostrar resultados
    st.markdown(f"**Constante aprendida por el perceptrón:** k = {learned_k:.4f}")
    plot_results(x, F, pred, k_real, learned_k)


st.title("📉 Perceptrón que aprende la Ley de Hooke")

st.markdown("""
Este dashboard muestra cómo un perceptrón simple puede aprender la ley de Hooke:  
\\( F = -k \\cdot x \\)  
Usamos datos sintéticos generados con una constante de resorte \\( k \\).
""")


