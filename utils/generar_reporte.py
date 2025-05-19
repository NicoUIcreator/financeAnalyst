import streamlit as st
import matplotlib.pyplot as plt

def mostrar_reporte_markdown(df, pred_30, pred_90, modelo_tipo, accuracy):
    st.markdown("## 🧾 Informe de Predicción BTC")
    
    st.markdown(f"""
**📅 Fecha de último dato disponible:** `{df.index[-1].date()}`  
**🧠 Modelo utilizado:** `{modelo_tipo}`  
**🎯 Accuracy del modelo sobre últimos 30 días:** `{accuracy:.2f}%`  
    """)

    st.markdown("### 📈 Gráfico de Predicción")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["price"], label="Histórico")
    ax.plot(pred_30.index, pred_30.values, label="Predicción 30 días")
    ax.plot(pred_90.index, pred_90.values, label="Predicción 90 días", linestyle="--")
    ax.set_title("Predicción del Precio BTC")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    st.markdown("### 📋 Resumen")
    st.markdown("""
- Los datos han sido normalizados y limpiados automáticamente.
- Se usó una ventana de 30 días para alimentar al modelo.
- Se generaron predicciones para 30 y 90 días hacia adelante.
    """)
