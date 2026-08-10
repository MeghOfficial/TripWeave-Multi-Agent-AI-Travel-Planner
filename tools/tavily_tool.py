from tavily import TavilyClient
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Create Tavily client using the API key
client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# Search the web using Tavily
def tavily_search(query):

    response = client.search(
        query=query,
        max_results=5
    )

    results = []

    # Process each search result
    for i, r in enumerate(response["results"], 1):

        title = r.get("title", "Unknown")
        url = r.get("url", "")
        snippet = r.get("content", "").strip()

        # Keep the text short
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        # Format the result
        results.append(
            f"{i}. **{title}**\n"
            f"   {url}\n"
            f"   {snippet}"
        )

    # Return all results as one string
    return "\n\n".join(results)