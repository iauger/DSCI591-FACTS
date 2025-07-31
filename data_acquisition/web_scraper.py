import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
import asyncio
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
import re
import time
from readability import Document
from urllib.parse import urlparse, unquote
import wikipediaapi



class WebScraper:
    def __init__(self, retries=5, backoff_factor=0.2, timeout=10, headers=None, delay=1, overrides=None, wiki=None):
        self.session = self._requests_session(retries, backoff_factor)
        self.timeout = timeout
        self.delay = delay
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                        '(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }
        self.url_overrides = overrides or {
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Food_history": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_arts_and_culture#Food_and_drink_history",
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Human_body_and_health": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_science,_technology,_and_mathematics#Human_body_and_health",
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Food_and_cooking": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_arts_and_culture#Food_and_cooking",
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Law,_crime,_and_military": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_arts_and_culture#Law,_crime,_and_military",
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Brain": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_science,_technology,_and_mathematics#Brain",
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Physics": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_science,_technology,_and_mathematics#Physics",
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Early_modern": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_history#Early_modern",
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Astronomy_and_spaceflight": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_science,_technology,_and_mathematics#Astronomy_and_spaceflight",
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Invertebrates": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_science,_technology,_and_mathematics#Invertebrates",            
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Modern": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_history#Modern",
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Disease": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_science,_technology,_and_mathematics#Disease_and_preventive_healthcare",
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Skin_and_hair": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_science,_technology,_and_mathematics#Skin_and_hair",
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Microwave_ovens": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_arts_and_culture#Microwave_ovens",
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Vertebrates": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_science,_technology,_and_mathematics#Vertebrates",
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Nutrition,_food,_and_drink": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_science,_technology,_and_mathematics#Nutrition,_food,_and_drink",
            "https://en.wikipedia.org/wiki/List_of_common_misconceptions#Music": "https://en.wikipedia.org/wiki/List_of_common_misconceptions_about_arts_and_culture#Music",
        
        }
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            user_agent='DSCI591-FACTS WebScraper (ian.auger@drexel.edu)'
        )

    
    def resolve_url(self, url):
        return self.url_overrides.get(url, url)
    
    def extract_page_title(self, url):
        path = urlparse(url).path
        if path.startswith("/wiki/"):
            return unquote(path[len("/wiki/"):])
        return None
    
    def extract_anchor_fragment(self, url):
        return unquote(urlparse(url).fragment).replace("_", " ")

    def get_section_text(self, page, anchor):
        anchor = anchor.lower()

        def search_sections(sections):
            for s in sections:
                if anchor in s.title.lower():
                    return s.text
                sub = search_sections(s.sections)
                if sub:
                    return sub
            return None

        return search_sections(page.sections)

    def get_wikipedia_text(self, url):
        title = self.extract_page_title(url)
        anchor = self.extract_anchor_fragment(url)

        if not title:
            return None, url

        page = self.wiki.page(title)
        if not page.exists():
            return None, url

        if anchor:
            section_text = self.get_section_text(page, anchor)
            if section_text:
                return section_text, url

        return page.text, url

    def _requests_session(self, retries, backoff_factor):
        session = requests.Session()
        retries_config = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount('http://', HTTPAdapter(max_retries=retries_config))
        session.mount('https://', HTTPAdapter(max_retries=retries_config))
        return session
    
    def fetch_page(self, url):
        try:
            response = self.session.get(url, timeout=self.timeout, headers=self.headers, allow_redirects=True)
            response.raise_for_status()
            time.sleep(self.delay)
            final_url = response.url

            if url != final_url:
                logging.info(f"Redirected: {url} → {final_url}")

            return response.text  
        except requests.RequestException as e:
            logging.error(f"[fetch_page] Error fetching {url}: {e}")
            return None
    
    def scrape_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        elements = soup.find_all(['p', 'li'])
        text = ' '.join(el.get_text(strip=True) for el in elements if el.get_text(strip=True))

        return text
    
    def extract_main_text(self, html_content):
        try:
            doc = Document(html_content)
            summary_html = doc.summary()
            soup = BeautifulSoup(summary_html, 'html.parser')
            return ' '.join(p.get_text().strip() for p in soup.find_all('p'))
        except Exception as e:
            logging.warning(f"Readability fallback: {e}")
            return self.scrape_html(html_content)
    
    async def scrape_dynamic_url(self, url):
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url)
                content = await page.content()
                await browser.close()
                return content
        except Exception as e:
            logging.error(f"[scrape_dynamic_url] Error fetching {url}: {e}")
            return None

    def clean_text(self, text):
        # Remove excessive whitespace, non-ASCII characters, etc.
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text.strip()
    
    def filter_extracted_text(self, text, max_len=2000):
        lines = text.splitlines()
        good_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip common comment noise or ads
            if any(bad in line.lower() for bad in ['comment', 'subscribe', 'wonder friend', 'click here', 'email']):
                continue
            if len(line) < 40:  # skip short lines that are likely nav or usernames
                continue
            good_lines.append(line)
            if sum(len(x) for x in good_lines) > max_len:
                break

        return ' '.join(good_lines)

    
    async def scrape_and_clean(self, url):
        resolved_url = self.resolve_url(url)
        
        if isinstance(resolved_url, str) and "wikipedia.org" in resolved_url:
            cleaned, resolved_url = self.get_wikipedia_text(resolved_url)
            return cleaned, resolved_url

        html_content = self.fetch_page(resolved_url)
        raw_text = self.extract_main_text(html_content) if html_content else None

        if not raw_text or len(raw_text) < 100:
            raw_text = self.scrape_dynamic_url(url)
        
        cleaned = self.clean_text(raw_text) if raw_text else None

        if cleaned:
            return self.filter_extracted_text(cleaned), resolved_url
        else:
            return None, resolved_url
    
    async def extract_multiple_sources(self, source_str):
        texts, confirmed_urls = [], []

        for url in str(source_str).split(";"):
            url = url.strip()
            if not url.startswith("http"):
                continue
            try:
                text, confirmed = await self.scrape_and_clean(url)
                if text:
                    texts.append(text)
                    confirmed_urls.append(confirmed)
            except Exception as e:
                logging.warning(f"Error processing {url}: {e}")
        
        return "\n\n".join(texts) if texts else None, "; ".join(confirmed_urls) if confirmed_urls else None
    
    async def augment_dataset(self, df):
        source_texts = []
        confirmed_urls = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scraping URLs"):
            url = row.get('Source')
            try:
                if pd.notna(url):
                    text, confirmed = await self.extract_multiple_sources(url)
                    source_texts.append(text)
                    confirmed_urls.append(confirmed)
                else:
                    source_texts.append(None)
                    confirmed_urls.append(None)
            except Exception as e:
                logging.error(f"[augment_dataset] Failed to process row {idx}: {e}")
                source_texts.append(None)
                confirmed_urls.append(None)

        df['source_text'] = source_texts
        df['confirmed_url'] = confirmed_urls
        return df