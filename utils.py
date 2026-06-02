import time
import os
import numpy as np
import pandas as pd
import yfinance as yf
import google.generativeai as genai
from newsapi import NewsApiClient
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

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
    def predict(self, date: datetime, verbose: bool = False) -> float:
        stock_history_data = self._get_stock_history_data(date)
        stock_news_titles = self._get_stock_news_titles(date)
        inputs = self.template.format(stock_history_data=stock_history_data, stock_news_titles=stock_news_titles)
        if verbose:
            print(inputs)
        max_retries = 5

        for attempt in range(max_retries):
            try:
                response = model.generate_content(inputs)
                return float(response.text.strip())
            except Exception as e:
                print(f"Retrying... {attempt + 1}/{max_retries}")
                print("Actual error:", repr(e))
                time.sleep(2)

        raise RuntimeError("Maximum retries exceeded")
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
    def backtesting(self, start_date: datetime, end_date: datetime, verbose: bool = False) -> pd.DataFrame:
        stock_history_data = yf.download(
            self.config['stock_symbol'],
            start=start_date,
            end=end_date + timedelta(days=1),
            auto_adjust=True,
            progress=False
        )

        if stock_history_data.empty:
            raise ValueError(
                f"No historical data returned for '{self.config['stock_symbol']}'. "
                "This is likely an SSL certificate issue. Run:\n"
                "  /Applications/Python\\ 3.13/Install\\ Certificates.command\n"
                "or:  pip install --upgrade certifi"
            )

        # Flatten MultiIndex columns if present (yfinance ≥0.2.x)
        if isinstance(stock_history_data.columns, pd.MultiIndex):
            stock_history_data.columns = stock_history_data.columns.get_level_values(0)

        stock_history_data.reset_index(inplace=True)

        # Normalise the date column name (may be 'Date' or 'Datetime')
        date_col = 'Date' if 'Date' in stock_history_data.columns else stock_history_data.columns[0]

        results = []
        for i, date in enumerate(stock_history_data[date_col]):
            actual_price = float(stock_history_data['Close'].iloc[i])
            predicted_price = self.predict(date, verbose)
            results.append({
                'Date': date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                'Predicted Price': predicted_price,
                'Actual Price': actual_price
            })

        results_df = pd.DataFrame(results)
        actual_prices = results_df['Actual Price'].dropna().values
        predicted_prices = results_df['Predicted Price'].dropna().values

        mse  = mean_squared_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(actual_prices, predicted_prices)
        r2   = r2_score(actual_prices, predicted_prices)
        ndei = rmse / np.std(actual_prices)

        print(f"\nMSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        print(f"NDEI: {ndei:.4f}")

        plt.figure(figsize=(12, 6))
        plt.plot(results_df['Date'], results_df['Predicted Price'], label='Predicted', marker='o')
        plt.plot(results_df['Date'], results_df['Actual Price'],    label='Actual',    marker='x')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Predicted vs Actual Stock Prices')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return results_df
    