import torch
import matplotlib.pyplot as plt
import streamlit as st

def generate_hooke_data(k, n=100):
    x = torch.linspace(-10, 10, n).unsqueeze(1)
    F = -k * x
    return x, F

def plot_results(x, F_real, F_pred, k_real, k_learned):
    fig, ax = plt.subplots()
    ax.plot(x.numpy(), F_real.numpy(), label=f'Real: k={k_real:.2f}')
    ax.plot(x.numpy(), F_pred.numpy(), '--', label=f'Perceptrón: k={k_learned:.2f}')
    ax.set_xlabel("Elongación x (m)")
    ax.set_ylabel("Fuerza F (N)")
    ax.set_title("Aprendizaje de la Ley de Hooke")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
