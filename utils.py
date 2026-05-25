import time
import numpy as np
import pandas as pd
import yfinance as yf
import google.generativeai as genai
from newsapi import NewsApiClient
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

genai.configure(api_key='gemini_api_key_here')

model = genai.GenerativeModel("gemini-2.5-flash")

class Agent():
    def __init__(self, config: dict):
        self.config = config
        genai.configure(api_key=config['genai_api_key'])
        self.llm = genai.GenerativeModel(model_name=config['model_name'])
        self.newsapi = NewsApiClient(api_key=config['news_api_key'])
        self.template = (
            "You are a financial analyst assistant with expertise in stock market analysis.\n"
            "Below is the historical stock price data and recent news headlines related to the stock.\n\n"
            "Historical Stock Data:\n"
            "{stock_history_data}\n\n"
            "Recent News Headlines:\n"
            "{stock_news_titles}\n\n"
            "Based on the above historical data and news sentiment, predict the closing stock price "
            "for the next trading day.\n"
            "Respond with only a single numeric value representing the predicted price. "
            "Do not include any explanation, units, or extra text."
        )