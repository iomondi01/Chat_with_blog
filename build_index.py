# build_index.py — builds your blog database

import os
import time
import requests
import chromadb
from bs4 import BeautifulSoup
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
BLOG_URL = os.getenv("BLOG_URL")  # WordPress.com blog URL

if not BLOG_URL:
    raise ValueError("❌ BLOG_URL not set. Please define it in your .env file like BLOG_URL=https://iomondi.wordpress.com/")

print("Using BLOG_URL:", BLOG_URL)

# Initialize Mistral and ChromaDB
client = Mistral(api_key=MISTRAL_API_KEY)
chroma_client = chromadb.PersistentClient(path="chroma_blog_db")
collection = chroma_client.get_or_create_collection("blog_rag_collection")


def fetch_wordpress_posts(site_url, per_page=50, page=1):
    """
    Fetch posts from a WordPress.com blog using the public API.
    Returns a list of dictionaries with 'id' and 'text'.
    """
    api_url = f"https://public-api.wordpress.com/rest/v1.1/sites/{site_url.replace('https://', '').replace('http://', '')}/posts/?number={per_page}&page={page}"
    print(f"Fetching: {api_url}")

    response = requests.get(api_url)
    print("Status code:", response.status_code)
    print("Response text (first 200 chars):", response.text[:200])

    response.raise_for_status()

    try:
        data = response.json()
    except Exception as e:
        print("❌ Failed to parse JSON. Full response below:")
        print(response.text)
        raise e

    documents = []
    for post in data.get("posts", []):
        title = post.get("title", "")
        content = post.get("content", "")
        text = BeautifulSoup(content, "html.parser").get_text()
        documents.append({"id": str(post["ID"]), "text": f"{title}\n\n{text}"})

    print(f"✅ Retrieved {len(documents)} posts")
    return documents


def get_embeddings_batch(texts, batch_size=32, delay=5):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(3):
            try:
                response = client.embeddings.create(model="mistral-embed", inputs=batch)
                all_embeddings.extend([item.embedding for item in response.data])
                break
            except Exception as e:
                if "rate limit" in str(e).lower():
                    print(f"⚠️ Rate limit hit, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
    return all_embeddings


def build_index():
    # Fetch posts
    docs = fetch_wordpress_posts(BLOG_URL)

    if not docs:
        print("⚠️ No posts found. Exiting...")
        return

    texts = [d["text"] for d in docs]

    print("🔢 Generating embeddings...")
    embeds = get_embeddings_batch(texts)

    collection.upsert(
        ids=[d["id"] for d in docs],
        documents=texts,
        embeddings=embeds
    )
    print("✅ Blog embedded and stored successfully!")


if __name__ == "__main__":
    build_index()
