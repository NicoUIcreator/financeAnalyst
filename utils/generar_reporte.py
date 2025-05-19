from fpdf import FPDF
import matplotlib.pyplot as plt
import os

def generar_reporte_pdf(df, pred_30, pred_90, modelo_tipo, accuracy):
    # Crear carpeta si no existe
    os.makedirs("/mnt/data", exist_ok=True)

    # Guardar gráfico temporal
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["price"], label="Histórico")
    plt.plot(pred_30.index, pred_30.values, label="Predicción 30 días")
    plt.plot(pred_90.index, pred_90.values, label="Predicción 90 días", linestyle="--")
    plt.title("Predicción del Precio BTC")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    graph_path = "/mnt/data/prediccion_btc.png"
    plt.savefig(graph_path)
    plt.close()

    # Crear PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(40, 40, 40)

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Reporte de Predicción BTC", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Modelo utilizado: {modelo_tipo}", ln=True)
    pdf.cell(200, 10, f"Accuracy sobre los últimos 30 días: {accuracy:.2f}%", ln=True)

    pdf.ln(10)
    pdf.image(graph_path, x=10, w=190)

    # Guardar PDF
    pdf_path = "/mnt/data/reporte_prediccion_btc.pdf"
    pdf.output(pdf_path)

    return pdf_path
