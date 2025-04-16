# app.py

import streamlit as st
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
    st.pyplot(fig)

# Streamlit App
def main():
    st.set_page_config(page_title="🗳️ Political Narrative Analyzer", layout="wide")
    st.title("🗳️ Political Narrative Analyzer")
    st.markdown("Analyze sentiment and narrative trends in political news.")

    topic = st.text_input("Enter a political topic:", "US elections")
    num_articles = st.slider("Number of articles to fetch", 5, 20, 10)

    if st.button("🔍 Analyze"):
        with st.spinner("Scraping and analyzing articles..."):
            urls = fetch_google_news_links(topic, num_articles)
            data = [extract_article_data(url) for url in urls]
            data = [d for d in data if d is not None]

        if not data:
            st.error("No articles could be processed.")
            return

        df = pd.DataFrame(data)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", data=csv, file_name="political_news_analysis.csv", mime="text/csv")

        plot_sentiment_trend(df, topic)

        st.subheader("📰 Articles and Sentiment")
        for _, row in df.iterrows():
            with st.expander(row["title"]):
                st.write(f"**Date:** {row['date'].strftime('%Y-%m-%d')}")
                st.write(f"**Sentiment Score:** `{row['sentiment']:.2f}`")
                st.write(f"**Summary:** {row['summary']}")
                st.markdown(f"[Read more]({row['url']})")

        st.subheader("📍 Named Entities Summary")
        st.dataframe(df[["title", "people", "organizations", "locations"]])

if __name__ == "__main__":
    main()
