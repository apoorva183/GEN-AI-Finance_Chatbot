import os
from typing import Dict, Optional, TypedDict
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# Load API keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq Chat Model
chat = ChatGroq(api_key=groq_api_key)

# Define agent state structure
class AgentState(TypedDict):
    stock_symbol: str
    indicators: Optional[Dict]
    metrics: Optional[Dict]
    analysis: Optional[str]
    approved: Optional[bool]

# Fetch stock price data
def fetch_stock_data(state: AgentState) -> AgentState:
    stock = yf.Ticker(state["stock_symbol"])
    hist = stock.history(period="3mo")
    state["indicators"] = compute_indicators(hist)
    return state

# Compute technical indicators
def compute_indicators(df: pd.DataFrame) -> Dict:
    df["change"] = df["Close"].diff()
    window_length = 14
    gain = np.where(df["change"] > 0, df["change"], 0)
    loss = np.where(df["change"] < 0, -df["change"], 0)
    avg_gain = pd.Series(gain).rolling(window=window_length, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window_length, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df[["RSI", "MACD", "VWAP"]].iloc[-1].to_dict()

# Fetch financial metrics
def evaluate_financial_metrics(state: AgentState) -> AgentState:
    stock = yf.Ticker(state["stock_symbol"])
    financials = stock.info
    state["metrics"] = {
        "P/E Ratio": financials.get("trailingPE", "N/A"),
        "Debt-to-Equity": financials.get("debtToEquity", "N/A"),
        "Profit Margin": financials.get("profitMargins", "N/A"),
    }
    return state

# AI Analysis
def analyze_financials(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template("""
    Analyze the financial performance of {stock_symbol} using the following data:
    
    **Technical Indicators:**
    RSI: {RSI}
    MACD: {MACD}
    VWAP: {VWAP}
    
    **Financial Metrics:**
    P/E Ratio: {P/E Ratio}
    Debt-to-Equity: {Debt-to-Equity}
    Profit Margin: {Profit Margin}
    
    Provide insights on trends, risks, and opportunities.
    """)

    response = chat.invoke(prompt.format(stock_symbol=state["stock_symbol"], **state["indicators"], **state["metrics"]))
    state["analysis"] = response.content
    return state

# Human-in-the-loop verification
def human_verification(state: AgentState) -> AgentState:
    if "approved" not in st.session_state:
        st.session_state.approved = False

    if st.button("✅ Confirm Analysis"):
        st.session_state.approved = True

    state["approved"] = st.session_state.approved
    return state


# Define LangGraph pipeline
graph = StateGraph(AgentState)
graph.add_node("fetch_stock", fetch_stock_data)
graph.add_node("evaluate_metrics", evaluate_financial_metrics)
graph.add_node("analyze", analyze_financials)
graph.add_node("verify", human_verification)

graph.add_edge("fetch_stock", "evaluate_metrics")
graph.add_edge("evaluate_metrics", "analyze")
graph.add_edge("analyze", "verify")
graph.add_edge("verify", END)

graph.set_entry_point("fetch_stock")
workflow = graph.compile()

# Streamlit UI
def main():
    st.title("📈 Financial Analyst AI")
    stock_symbol = st.text_input("Enter Stock Symbol:", "AAPL")

    if st.button("Analyze Stock"):
        input_state = {"stock_symbol": stock_symbol}
        output_state = workflow.invoke(input_state)

        st.subheader("📊 Technical Indicators")
        st.json(output_state["indicators"])

        st.subheader("📈 Financial Metrics")
        st.json(output_state["metrics"])

        st.subheader("📝 AI-Generated Financial Analysis")
        st.markdown(output_state["analysis"])

        # Human verification
        if output_state["approved"]:
            st.success("✅ Analysis Confirmed!")

if __name__ == "__main__":
    main()
