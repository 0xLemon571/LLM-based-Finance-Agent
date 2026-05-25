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
    def _get_stock_history_data(self, date: datetime) -> pd.DataFrame:
        start_date = date - timedelta(days=self.config['days'])
        print("Ticker:", self.config['stock_symbol'])
        print("Start:", start_date)
        print("End:", date)
        stock_data = yf.download(
            self.config['stock_symbol'],
            start=start_date,
            end=date,
            auto_adjust=True,
            progress=False
        )
        print(stock_data.shape)
        print(stock_data.head())
        if stock_data.empty:
            raise ValueError(
                f"No data returned for symbol '{self.config['stock_symbol']}' between "
                f"{start_date.date()} and {date.date()}. "
                "Check your SSL certificates and that the symbol is valid."
            )
        return stock_data
    def _get_stock_news_titles(self, date: datetime) -> list:
        try:
            stock = yf.Ticker(self.config['stock_symbol'])
            stock_info = stock.info
            stock_name = stock_info.get('longName', self.config['stock_symbol'])
        except Exception:
            stock_name = self.config['stock_symbol']

        previous_date = date - timedelta(days=1)
        start_date = previous_date.strftime("%Y-%m-%d")
        end_date = date.strftime("%Y-%m-%d")

        try:
            all_articles = self.newsapi.get_everything(
                q=stock_name,
                from_param=start_date,
                to=end_date,
                language='en',
                sort_by='relevancy'
            )
            titles = [article['title'] for article in all_articles['articles']]
        except Exception as e:
            print(f"\nWarning: Could not fetch news ({e}). Proceeding without news.")
            titles = []
        return titles