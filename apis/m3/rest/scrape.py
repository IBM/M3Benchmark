import requests
import time

def scrape_metrics(url="http://localhost:8003/metrics", interval=5):
    while True:
        response = requests.get(url)
        print(response.text)
        print("-" * 50)
        time.sleep(interval)

scrape_metrics()