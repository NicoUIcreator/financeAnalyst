import streamlit as st
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task
import google.generativeai as genai
from pydantic import BaseModel
from tools.financial_tools import YFinanceStockTool

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de Gemini Pro
@st.cache_resource
def load_gemini_model():
    # Configura la clave de API de Gemini Pro
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('')
    return model

# Definir modelos Pydantic para salida estructurada
class StockAnalysis(BaseModel):
    symbol: str
    company_name: str
    current_price: float
    market_cap: float
    pe_ratio: float
    recommendation: str
    analysis_summary: str
    risk_assessment: str
    technical_indicators: dict
    fundamental_metrics: dict

# Crear agentes y tareas
def create_agents_and_tasks(symbol: str):
    # Cargar modelo Gemini Pro
    model = load_gemini_model()
    
    # Inicializar herramientas
    stock_tool = YFinanceStockTool()
    
    # Agente de Análisis de Acciones
    stock_analysis_agent = Agent(
        role="Wall Street Financial Analyst",
        goal=f"Conduct a comprehensive, data-driven analysis of {symbol} stock using real-time market data",
        backstory="""You are a seasoned Wall Street analyst with 15+ years of experience in equity research.
                     You're known for your meticulous analysis and data-driven insights.
                     You ALWAYS base your analysis on real-time market data, never relying solely on pre-existing knowledge.
                     You're an expert at interpreting financial metrics, market trends, and providing actionable insights.""",
        llm=model,
        verbose=True,
        memory=True,
        tools=[stock_tool]
    )

    # Agente de Escritura de Informes
    report_writer_agent = Agent(
        role="Financial Report Specialist",
        goal="Transform detailed financial analysis into a professional, comprehensive investment report",
        backstory="""You are an expert financial writer with a track record of creating institutional-grade research reports.
                     You excel at presenting complex financial data in a clear, structured format.
                     You always maintain professional standards while making reports accessible and actionable.
                     You're known for your clear data presentation, trend analysis, and risk assessment capabilities.""",
        llm=model,
        verbose=True
    )

    # Tarea de Análisis
    analysis_task = Task(
        description=f"""Analyze {symbol} stock using the stock_data_tool to fetch real-time data. Your analysis must include:

        1. Latest Trading Information (HIGHEST PRIORITY)
           - Latest stock price with specific date
           - Percentage change
           - Trading volume
           - Market status (open/closed)
           - Highlight if this is from the most recent trading session

        2. 52-Week Performance (CRITICAL)
           - 52-week high with exact date
           - 52-week low with exact date
           - Current price position relative to 52-week range
           - Calculate percentage from highs and lows

        3. Financial Deep Dive
           - Market capitalization
           - P/E ratio and other key metrics
           - Revenue growth and profit margins
           - Dividend information (if applicable)

        4. Technical Analysis
           - Recent price movements
           - Volume analysis
           - Key technical indicators

        5. Market Context
           - Business summary
           - Analyst recommendations
           - Key risk factors

        IMPORTANT: 
        - ALWAYS use the stock_data_tool to fetch real-time data
        - Begin your analysis with the latest price and 52-week data
        - Include specific dates for all price points
        - Clearly indicate when each price point was recorded
        - Calculate and show percentage changes
        - Verify all numbers with live data
        - Compare current metrics with historical trends""",
        expected_output="A comprehensive analysis report with real-time data, including all specified metrics and clear section breakdowns",
        agent=stock_analysis_agent
    )

    # Tarea de Informe
    report_task = Task(
        description=f"""Transform the analysis into a professional investment report for {symbol}. The report must:

        1. Structure:
           - Begin with an executive summary
           - Use clear section headers
           - Include tables for data presentation
           - Add emoji indicators for trends (📈 📉)

        2. Content Requirements:
           - Include timestamps for all data points
           - Present key metrics in tables
           - Use bullet points for key insights
           - Compare metrics to industry averages
           - Explain technical terms
           - Highlight potential risks

        3. Sections:
           - Executive Summary
           - Market Position Overview
           - Financial Metrics Analysis
           - Technical Analysis
           - Risk Assessment
           - Future Outlook

        4. Formatting:
           - Use markdown formatting
           - Create tables for data comparison
           - Include trend emojis
           - Use bold for key metrics
           - Add bullet points for key takeaways

        IMPORTANT:
        - Maintain professional tone
        - Clearly state all data sources
        - Include risk disclaimers
        - Format in clean, readable markdown""",
        expected_output="A professionally formatted investment report in markdown, with clear sections, data tables, and visual indicators",
        agent=report_writer_agent
    )

    # Crear Crew
    crew = Crew(
        agents=[stock_analysis_agent, report_writer_agent],
        tasks=[analysis_task, report_task],
        process=Process.sequential,
        verbose=True
    )

    return crew

# Interfaz de usuario de Streamlit
st.set_page_config(page_title="Multi-Agent AI Financial Analyst", layout="wide")

st.title("🎯 Multi-Agent AI Financial Analyst")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Entrada de clave de API
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Enter your Gemini API key"
    )
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    # Entrada de símbolo de acción
    symbol = st.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter a stock symbol (e.g., AAPL, GOOGL)"
    ).upper()

    # Botón de análisis
    analyze_button = st.button("Analyze Stock", type="primary")

# Contenido principal
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.report = None

if analyze_button:
    try:
        with st.spinner(f'Analyzing {symbol}... This may take a few minutes.'):
            # Crear y ejecutar el crew
            crew = create_agents_and_tasks(symbol)
            result = crew.kickoff()
            # Convertir el resultado a string si es necesario
            if hasattr(result, 'raw'):
                st.session_state.report = result.raw
            else:
                st.session_state.report = str(result)
            st.session_state.analysis_complete = True

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if st.session_state.analysis_complete and st.session_state.report:
    st.markdown("### Analysis Report")
    st.markdown(st.session_state.report)
    
    # Botón de descarga
    st.download_button(
        label="Download Report",
        data=st.session_state.report,
        file_name=f"stock_analysis_{symbol}_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown"
    )

# Pie de página
st.markdown("---")