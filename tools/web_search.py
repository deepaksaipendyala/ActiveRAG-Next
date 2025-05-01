# tools/web_search.py

import requests
from typing import List
from utils.config import settings

def search_web(query: str, k: int = 3) -> List[str]:
    """
    Search the web using Tavily API and return top-k snippets.

    Args:
        query (str): Search query.
        k (int): Number of top snippets to return.

    Returns:
        List[str]: List of retrieved snippets.
    """
    api_key = settings.TAVILY_API_KEY
    url = "https://api.tavily.com/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "query": query,
        "num_results": k,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        snippets = [res["content"] for res in data.get("results", []) if "content" in res]
        return snippets[:k]
    
    except Exception as e:
        print(f"[Web Search] Error: {e}")
        return []
