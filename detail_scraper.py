import os
import csv
import json
import time
import re
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# ---- CONFIG ----
URL_CACHE_FILE = "data/listing_urls.json"                 # input: list of detail-page URLs
CSV_NAME       = "data/automarket_autos.csv"              # output CSV
DELAY_SECONDS  = 2.0                                 # polite delay between detail requests
TIMEOUT        = (10, 30)                            # (connect, read) timeouts

# Output columns (Slovak site field names to keep your later pipeline stable)
FIELDS = [
    "Title", "Cena",
    "Rok výroby","Stav","Najazdené",
    "Palivo","Objem","Výkon",
    "Prevodovka","Karoséria","Pohon",
    "Farba","Norma",
    "SellerType",
    "URL","URL_canon"
]

# ---- HTTP session with retries ----
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "sk-SK,sk;q=0.9,en;q=0.8"
})
retry = Retry(
    total=3,
    backoff_factor=0.7,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ---- Robust price matching helpers ----
PRICE_ONLY_RE = re.compile(r'^\s*\d[\d\s\.,]*\s*€\s*$', re.I)

def extract_price_from_soup(soup: BeautifulSoup) -> str:
    """
    Returns the main ad price text like '30 750 €' or 'Cena: Dohodou'.
    Strategy:
      0) meta itemprop=price (microdata) if present
      1) common CSS selectors used across variants
      2) headings that contain only the amount
      3) worded cases like 'Cena: Dohodou' / 'Na vyžiadanie'
      4) last-ditch: any '... €' line that doesn't look like fees/DPH
    """
    # 0) microdata (sometimes present)
    meta = soup.find("meta", attrs={"itemprop": "price"})
    if meta and meta.get("content"):
        content = meta["content"].strip()
        # normalize numeric-like content to '<amount> €'
        if re.search(r'^\d[\d\s\.,]*$', content):
            return f"{content} €"
        return content

    # 1) Try known selectors
    for css in [
        "h2.p-amount",
        ".p-amount",
        ".price .amount",
        ".detail__price",
        ".price-box .amount",
        ".price",
    ]:
        el = soup.select_one(css)
        if el:
            txt = el.get_text(" ", strip=True)
            if "€" in txt or re.search(r"\d", txt):
                return txt

    # 2) Headings with just the amount (covers <h1><strong>1 000 €</strong></h1>)
    for tag in ["h1", "h2", "h3"]:
        for el in soup.find_all(tag):
            txt = el.get_text(" ", strip=True)
            if PRICE_ONLY_RE.match(txt):
                return txt

    # 3) Worded cases
    worded = soup.find(string=re.compile(r'Cena\s*:\s*(Dohodou|Na vyžiadanie)', re.I))
    if worded:
        return worded.strip()

    # 4) Last-ditch: any amount with €, but avoid typical fee lines
    any_amount = soup.find(string=re.compile(r'\d[\d\s\.,]*\s*€'))
    if any_amount:
        txt = any_amount.strip()
        if not re.search(r'(DPH|Registračný|Poplatok|odpočet)', txt, re.I):
            return txt

    return ""


# ---- helpers ----
def extract_title_from_url(url: str) -> str:
    """
    Build a human-ish title from the URL slug:
    https://www.autobazar.sk/<id>/<slug>/ -> "Skoda Superb Combi 2 0 Tdi ..."
    """
    try:
        parts = urlparse(url).path.strip("/").split("/")
        # detail URL format is usually /<id>/<slug>
        if len(parts) >= 2:
            slug = parts[1]
        else:
            slug = parts[-1] if parts else ""
        if not slug:
            return ""
        return slug.replace("-", " ").title()
    except Exception:
        return ""

def parse_ad_details(url: str) -> dict:
    """
    Fetch a detail page and extract all fields we need.
    """
    resp = session.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    data = {f: "" for f in FIELDS}
    data["URL"] = url

    # Canonical URL (helps dedupe/merging later)
    can = soup.find("link", rel="canonical")
    data["URL_canon"] = (can["href"].strip() if can and can.has_attr("href") else url)

    # Title – generate from URL slug (fast & stable)
    data["Title"] = extract_title_from_url(data["URL_canon"] or url)

    # Price ("Cena") – robust extractor
    data["Cena"] = extract_price_from_soup(soup)

    # Parameters table
    for row in soup.select("div.ab-grid.parameters__row"):
        lbl = row.select_one("span.parameters__label")
        val = row.select_one("p.parameters__value")
        if not (lbl and val):
            continue
        key = lbl.get_text(strip=True).rstrip(":")
        text = val.get_text(" ", strip=True).replace("Overiť", "").strip()
        if key in data:
            data[key] = text

    # SellerType (firma vs súkromný predajca)
    # 1) textual hint
    p_txt = soup.find("p", string=lambda t: t and ("Firma" in t or "Súkromný predajca" in t))
    if p_txt:
        txt = p_txt.get_text(" ", strip=True)
        data["SellerType"] = "Firma" if "Firma" in txt else "Súkromný predajca"
    else:
        # 2) icon fallback
        icon = soup.select_one("h2.seller__title .icon_firm, h2.seller__title .icon_anonym")
        if icon and "icon_firm" in icon.get("class", []):
            data["SellerType"] = "Firma"
        elif icon and "icon_anonym" in icon.get("class", []):
            data["SellerType"] = "Súkromný predajca"
        else:
            data["SellerType"] = ""  # unknown

    return data

def load_url_list(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        urls = json.load(f)
    # ensure unique and stable order
    seen, deduped = set(), []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped

def load_already_scraped(csv_path: str) -> set:
    """
    If CSV exists, read its URL_canon (or URL) to skip already-scraped ads.
    """
    if not os.path.exists(csv_path):
        return set()
    scraped = set()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        # prefer canonical if present, else raw URL
        key = "URL_canon" if "URL_canon" in reader.fieldnames else ("URL_canonical" if "URL_canonical" in reader.fieldnames else "URL")
        for row in reader:
            u = (row.get(key) or row.get("URL") or "").strip()
            if u:
                scraped.add(u)
    return scraped

def main():
    # Load all URLs
    all_links = load_url_list(URL_CACHE_FILE)
    print(f"Loaded {len(all_links)} URLs from {URL_CACHE_FILE}")

    # Prepare CSV (append if exists to support resume)
    file_exists = os.path.exists(CSV_NAME)
    scraped = load_already_scraped(CSV_NAME)
    print(f"Found {len(scraped)} already-scraped rows in {CSV_NAME}" if file_exists else "No existing CSV, will create a new one.")

    written = 0
    with open(CSV_NAME, "a", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()

        for idx, url in enumerate(all_links, 1):
            # skip if we've already scraped canonical URL or the raw URL
            if url in scraped:
                continue

            print(f"[{idx}/{len(all_links)}] Scraping {url}")
            try:
                rec = parse_ad_details(url)

                # if canonical known and already scraped, skip
                canon = rec.get("URL_canon") or ""
                if canon and canon in scraped:
                    continue

                writer.writerow(rec)
                fp.flush()
                written += 1

                # mark both forms as scraped for this run
                if canon:
                    scraped.add(canon)
                scraped.add(url)

            except Exception as e:
                print(f"  Error on {url}: {e}")

            time.sleep(DELAY_SECONDS)

    print(f"Done – wrote {written} new rows to {CSV_NAME}")

if __name__ == "__main__":
    main()
