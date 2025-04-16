!pip install newspaper3k vaderSentiment spacy beautifulsoup4 pandas matplotlib lxml ipywidgets
!python -m spacy download en_core_web_sm

# app_colab.py

import requests
from bs4 import BeautifulSoup
from newspaper import Article
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from datetime import datetime

# Setup
analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

# Google News Scraper
def fetch_google_news_links(topic, num_results=10):
    search_url = f"https://news.google.com/search?q={topic.replace(' ', '%20')}%20politics&hl=en-US&gl=US&ceid=US:en"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('article')[:num_results]

    urls = []
    for article in articles:
        a_tag = article.find('a', href=True)
        if a_tag:
            relative_link = a_tag['href']
            if relative_link.startswith("./articles/"):
                full_link = "https://news.google.com" + relative_link[1:]
                urls.append(full_link)
    return urls

# Article Parser
def extract_article_data(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        date = article.publish_date or datetime.now()

        doc = nlp(article.text)
        people = set([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
        orgs = set([ent.text for ent in doc.ents if ent.label_ == "ORG"])
        gpes = set([ent.text for ent in doc.ents if ent.label_ == "GPE"])

        return {
            "title": article.title,
            "summary": article.summary,
            "text": article.text,
            "url": url,
            "date": date,
            "sentiment": analyzer.polarity_scores(article.summary)["compound"],
            "people": ", ".join(people),
            "organizations": ", ".join(orgs),
            "locations": ", ".join(gpes),
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

# Trend Plot
def plot_sentiment_trend(df, topic):
    df = df.dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    grouped = df.groupby("date")["sentiment"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grouped["date"], grouped["sentiment"], marker='o')
    ax.set_title(f"Sentiment Trend for '{topic}'")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Sentiment")
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.show()

# Main Function for Colab
def main():
    import ipywidgets as widgets
    from IPython.display import display
    
    # User Input for Topic
    topic_input = widgets.Text(value='US elections', description='Topic:')
    num_articles_input = widgets.IntSlider(value=10, min=5, max=20, step=1, description='Articles:')
    
    display(topic_input, num_articles_input)
    
    def on_button_click(b):
        topic = topic_input.value
        num_articles = num_articles_input.value
        
        print(f"Fetching news articles about: {topic}")
        
        # Fetch URLs and analyze articles
        urls = fetch_google_news_links(topic, num_articles)
        data = [extract_article_data(url) for url in urls]
        data = [d for d in data if d is not None]
        
        if not data:
            print("No articles could be processed.")
            return

        # Create DataFrame for Display
        df = pd.DataFrame(data)
        
        # Plot Sentiment Trend
        plot_sentiment_trend(df, topic)

        # Display Articles and Sentiment Analysis
        print("\n📰 Articles and Sentiment:")
        for _, row in df.iterrows():
            print(f"\nTitle: {row['title']}")
            print(f"Date: {row['date'].strftime('%Y-%m-%d')}")
            print(f"Sentiment Score: {row['sentiment']:.2f}")
            print(f"Summary: {row['summary']}")
            print(f"Read more: {row['url']}")
        
        # Named Entities Summary
        print("\n📍 Named Entities Summary:")
        print(df[["title", "people", "organizations", "locations"]])

    # Button to trigger analysis
    button = widgets.Button(description="🔍 Analyze")
    button.on_click(on_button_click)
    display(button)

# Run the function in Colab
main()
