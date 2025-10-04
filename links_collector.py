import requests
import json
import time
from requests.exceptions import HTTPError

BASE_SEARCH    = "https://osobne-auta.autobazar.sk"
ORDER_PARAM    = "23"
MAX_PAGES      = 2000
DELAY          = 2.0
URL_CACHE_FILE = "data/listing_urls.json"

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

def get_listing_links(page: int):
    """Fetch all detail-URLs from one listing page."""
    if page == 1:
        url = f"{BASE_SEARCH}/?p[order]={ORDER_PARAM}"
    else:
        url = f"{BASE_SEARCH}/?p[page]={page}&p[order]={ORDER_PARAM}"
    r = session.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for card in soup.select("div.item"):
        a = card.select_one("a[href]")
        if a:
            href = a["href"]
            full = href if href.startswith("http") else "https://www.autobazar.sk" + href
            links.append(full)
    print(f"[Page {page}] Found {len(links)} ads")
    return links

def main():
    try:
        with open(URL_CACHE_FILE, "r", encoding="utf-8") as f:
            all_links = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_links = []

    for p in range(1, MAX_PAGES + 1):
        try:
            new = get_listing_links(p)
        except HTTPError as e:
            if e.response.status_code == 404:
                print("No more pages. Stopping.")
                break
            else:
                raise

        all_links.extend(new)
        all_links = list(dict.fromkeys(all_links))

        with open(URL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(all_links, f, ensure_ascii=False, indent=2)

        time.sleep(DELAY)

    print(f"Total unique URLs saved: {len(all_links)}")

if __name__ == "__main__":
    from bs4 import BeautifulSoup
    main()
