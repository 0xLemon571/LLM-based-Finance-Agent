import ssl
import certifi
import os

# Must patch SSL BEFORE any network-related imports
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import json
from datetime import datetime
from utils import Agent

def main():
    with open('config.json', 'r', encoding="utf-8") as file:
        config = json.load(file)
    agent = Agent(config)
    start_date = datetime(2026, 5, 1)
    end_date = datetime(2026, 6, 1)
    agent.backtesting(start_date, end_date, verbose=False)

if __name__ == '__main__':
    main()