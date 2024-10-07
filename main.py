from bs4 import BeautifulSoup
import requests
import joblib
from pprint import pprint

url = "https://www.coindesk.com/"

# Scrape coindesk website for finanacial news
response = requests.get(url)
page = response.text
soup = BeautifulSoup(page, "html.parser")

# Load the model and vectorizer
loaded_model = joblib.load("text_classification_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")


article_data = {}

latest_news = list(soup.find(name="div", class_="kuxwiI"))
for news in range(len(latest_news)):
    article_title = latest_news[news].find(class_="card-title").getText()
    article_summary = latest_news[news].find(class_="card-descriptionstyles__CardDescriptionWrapper-sc-463n0d-0").getText()
    article_link = url + latest_news[news].find(class_="card-title-link").get("href")

    # Category prediction
    vectorized_text = vectorizer.transform([article_title + article_summary])
    article_category = loaded_model.predict(vectorized_text)

    article_data[news] = {"article_category": article_category[0], "article_title": article_title, "article_summary": article_summary, "article_link": article_link}

pprint(article_data)