#!/usr/bin/env python3
"""
Web Search Agent for Ollama AI Assistant

A robust web search tool that handles rate limiting and provides
clean search results for AI assistants.
"""

import requests
import time
import random
from typing import List, Dict, Optional
from urllib.parse import quote_plus, urljoin
from bs4 import BeautifulSoup
import json
import logging
from dataclasses import dataclass
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Data class for search results"""
    title: str
    url: str
    snippet: str
    source: str = "web"

class WebSearchAgent:
    """
    A robust web search agent that uses multiple search engines
    and implements rate limiting to avoid API restrictions.
    """
    
    def __init__(self):
        self.session = requests.Session()
        
        # Rotate user agents to avoid blocking
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]
        
        self._update_headers()
        
        # Rate limiting parameters
        self.last_request_time = 0
        self.min_delay = 3  # Minimum seconds between requests
        self.max_delay = 7  # Maximum random delay
        self.retry_attempts = 3
        self.retry_delay = 8
        
        # Search engines configuration - adding DuckDuckGo as primary since it's less likely to block
        self.search_engines = [
            {
                'name': 'duckduckgo',
                'url': 'https://html.duckduckgo.com/html/',
                'parser': self._parse_duckduckgo_results
            },
            {
                'name': 'bing',
                'url': 'https://www.bing.com/search',
                'parser': self._parse_bing_results
            },
            {
                'name': 'google',
                'url': 'https://www.google.com/search',
                'parser': self._parse_google_results
            },
            {
                'name': 'searx_org',
                'url': 'https://searx.org/search',
                'parser': self._parse_searx_results
            },
            {
                'name': 'yandex',
                'url': 'https://yandex.com/search/',
                'parser': self._parse_yandex_results
            }
        ]
    
    def _update_headers(self):
        """Update session headers with random user agent"""
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last + random.uniform(1, self.max_delay - self.min_delay)
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict = None, timeout: int = 20) -> Optional[requests.Response]:
        """Make a robust HTTP request with error handling"""
        self._enforce_rate_limit()
        self._update_headers()  # Rotate user agent
        
        for attempt in range(self.retry_attempts):
            try:
                response = self.session.get(url, params=params, timeout=timeout, allow_redirects=True)
                
                logger.info(f"Response status: {response.status_code}, Content length: {len(response.text)}")
                
                if response.status_code == 200:
                    # Debug: Save response for troubleshooting
                    if len(response.text) < 1000:
                        logger.warning(f"Very short response: {response.text[:500]}")
                    return response
                elif response.status_code == 429:  # Rate limited
                    logger.warning(f"Rate limited (429), waiting longer...")
                    time.sleep(self.retry_delay * 2)
                elif response.status_code == 403:
                    logger.warning(f"Forbidden (403) - likely blocked by {url}")
                else:
                    logger.warning(f"HTTP {response.status_code} from {url}")
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.retry_attempts - 1:
                    sleep_time = self.retry_delay * (attempt + 1) + random.uniform(2, 5)
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
        
        logger.error(f"All {self.retry_attempts} attempts failed for URL: {url}")
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove common artifacts
        text = re.sub(r'\.\.\..*?›', '', text)
        text = re.sub(r'›.*?›', '', text)
        
        return text[:300]  # Limit snippet length
    
    def _parse_google_results(self, html_content: str) -> List[SearchResult]:
        """Parse search results from Google"""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Multiple selectors for Google results - updated for 2024/2025
            selectors = [
                'div.MjjYud',  # Current Google result container
                'div.g',       # Standard results
                'div.tF2Cxc',  # Alternative current selector
                'div[data-ved]',  # Fallback
                '.rc'          # Classic selector
            ]
            
            result_divs = []
            for selector in selectors:
                result_divs = soup.select(selector)
                if result_divs:
                    logger.info(f"Using selector '{selector}' for Google results")
                    break
            
            logger.info(f"Found {len(result_divs)} potential Google results")
            
            for div in result_divs[:8]:  # Get more to filter better
                try:
                    # Try multiple title selectors - updated for current Google
                    title_elem = (
                        div.select_one('h3') or 
                        div.select_one('.LC20lb') or
                        div.select_one('a h3') or
                        div.select_one('[role="heading"] h3') or
                        div.select_one('div[role="heading"]')
                    )
                    
                    if not title_elem:
                        continue
                    
                    title = self._clean_text(title_elem.get_text())
                    
                    # Find the link - improved selectors
                    link_elem = (
                        div.select_one('a[href^="http"]') or 
                        div.select_one('a[href^="/url?q=http"]') or
                        div.select_one('a') or
                        div.find('a', href=True)
                    )
                    
                    if not link_elem:
                        continue
                    
                    url = link_elem.get('href', '')
                    
                    # Clean Google redirect URLs
                    if url.startswith('/url?q='):
                        url = url.split('/url?q=')[1].split('&')[0]
                        from urllib.parse import unquote
                        url = unquote(url)
                    
                    # Skip Google's internal links
                    if 'google.com' in url or url.startswith('/search') or not url.startswith('http'):
                        continue
                    
                    # Find snippet - updated selectors
                    snippet_selectors = [
                        '.VwiC3b',     # Current Google snippet class
                        '.yXK7lf',     # Alternative current class
                        '.X5LH0c',     # Another current class  
                        '.s3v9rd',     # Alternative snippet class
                        '.st',         # Classic snippet class
                        'span[data-ved]',  # Fallback
                        '.IsZvec'      # Another fallback
                    ]
                    
                    snippet = ""
                    for selector in snippet_selectors:
                        snippet_elem = div.select_one(selector)
                        if snippet_elem:
                            snippet = self._clean_text(snippet_elem.get_text())
                            break
                    
                    if title and url and len(results) < 6:
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source="google"
                        ))
                
                except Exception as e:
                    logger.debug(f"Error parsing Google result: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing Google results: {e}")
        
    def _parse_duckduckgo_results(self, html_content: str) -> List[SearchResult]:
        """Parse search results from DuckDuckGo HTML version"""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # DuckDuckGo HTML selectors
            result_divs = soup.select('.result')
            
            logger.info(f"Found {len(result_divs)} DuckDuckGo results")
            
            for div in result_divs[:6]:
                try:
                    # Extract title and URL
                    title_link = div.select_one('.result__a')
                    if not title_link:
                        continue
                    
                    title = self._clean_text(title_link.get_text())
                    url = title_link.get('href', '')
                    
                    # Extract snippet
                    snippet_elem = div.select_one('.result__snippet')
                    snippet = self._clean_text(snippet_elem.get_text()) if snippet_elem else ""
                    
                    if title and url:
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source="duckduckgo"
                        ))
                
                except Exception as e:
                    logger.debug(f"Error parsing DuckDuckGo result: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing DuckDuckGo results: {e}")
        
        return results
    
    def _parse_yandex_results(self, html_content: str) -> List[SearchResult]:
        """Parse search results from Yandex"""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Yandex result selectors
            result_divs = soup.select('.serp-item')
            
            logger.info(f"Found {len(result_divs)} Yandex results")
            
            for div in result_divs[:6]:
                try:
                    # Extract title and URL
                    title_link = div.select_one('.organic__url-text') or div.select_one('h2 a')
                    if not title_link:
                        continue
                    
                    title = self._clean_text(title_link.get_text())
                    url = title_link.get('href', '')
                    
                    # Extract snippet
                    snippet_elem = div.select_one('.organic__text')
                    snippet = self._clean_text(snippet_elem.get_text()) if snippet_elem else ""
                    
                    if title and url:
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source="yandex"
                        ))
                
                except Exception as e:
                    logger.debug(f"Error parsing Yandex result: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing Yandex results: {e}")
        
        return results
    
    def _parse_bing_results(self, html_content: str) -> List[SearchResult]:
        """Parse search results from Bing"""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Debug: Check if we have a valid Bing page
            if 'bing.com' not in html_content.lower() and 'microsoft' not in html_content.lower():
                logger.warning("Response doesn't appear to be from Bing")
                # Save a sample for debugging
                logger.debug(f"Sample response: {html_content[:500]}")
            
            # Multiple Bing result selectors
            selectors = [
                '.b_algo',           # Primary Bing results
                'li.b_algo',         # Alternative
                '.b_searchResult',   # Fallback
                'ol#b_results li'    # Direct path
            ]
            
            result_divs = []
            for selector in selectors:
                result_divs = soup.select(selector)
                if result_divs:
                    logger.info(f"Using Bing selector '{selector}' - found {len(result_divs)} results")
                    break
            
            if not result_divs:
                logger.warning("No Bing results found with any selector")
                # Try to find any links as fallback
                all_links = soup.find_all('a', href=True)
                logger.info(f"Found {len(all_links)} total links in page")
            
            logger.info(f"Found {len(result_divs)} Bing results")
            
            for div in result_divs[:6]:
                try:
                    # Extract title and URL
                    title_selectors = [
                        'h2 a',
                        'h2',
                        '.b_topTitle a',
                        'a[href]'
                    ]
                    
                    title_link = None
                    for selector in title_selectors:
                        title_link = div.select_one(selector)
                        if title_link:
                            break
                    
                    if not title_link:
                        continue
                    
                    title = self._clean_text(title_link.get_text())
                    url = title_link.get('href', '')
                    
                    # Extract snippet
                    snippet_selectors = [
                        '.b_caption p',
                        '.b_caption',
                        '.b_snippetText',
                        'p'
                    ]
                    
                    snippet = ""
                    for selector in snippet_selectors:
                        snippet_elem = div.select_one(selector)
                        if snippet_elem and snippet_elem != title_link.parent:
                            snippet = self._clean_text(snippet_elem.get_text())
                            break
                    
                    if title and url:
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source="bing"
                        ))
                
                except Exception as e:
                    logger.debug(f"Error parsing Bing result: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing Bing results: {e}")
        
        return results
    
    def _parse_searx_results(self, html_content: str) -> List[SearchResult]:
        """Parse search results from SearX instances"""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Multiple selectors for different SearX versions
            selectors = [
                'div.result',
                'article.result',
                '.result',
                'div[class*="result"]'
            ]
            
            result_divs = []
            for selector in selectors:
                result_divs = soup.select(selector)
                if result_divs:
                    break
            
            logger.info(f"Found {len(result_divs)} SearX results")
            
            for div in result_divs[:6]:
                try:
                    # Extract title and URL
                    title_link = (
                        div.select_one('h3 a') or 
                        div.select_one('h4 a') or 
                        div.select_one('a[href^="http"]')
                    )
                    
                    if not title_link:
                        continue
                    
                    title = self._clean_text(title_link.get_text())
                    url = title_link.get('href', '')
                    
                    # Extract snippet
                    snippet_selectors = [
                        'p.content',
                        '.content',
                        'p',
                        'div.content'
                    ]
                    
                    snippet = ""
                    for selector in snippet_selectors:
                        snippet_elem = div.select_one(selector)
                        if snippet_elem and snippet_elem != title_link.parent:
                            snippet = self._clean_text(snippet_elem.get_text())
                            break
                    
                    if title and url:
                        results.append(SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source="searx"
                        ))
                
                except Exception as e:
                    logger.debug(f"Error parsing SearX result: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing SearX results: {e}")
        
        return results
    
    def _search_with_engine(self, query: str, engine_config: Dict) -> List[SearchResult]:
        """Search using a specific search engine"""
        try:
            if engine_config['name'] == 'google':
                params = {
                    'q': query,
                    'num': 10,
                    'hl': 'en'
                }
            elif engine_config['name'] == 'bing':
                params = {
                    'q': query,
                    'count': 10
                }
            elif engine_config['name'] == 'duckduckgo':
                params = {
                    'q': query
                }
            elif engine_config['name'] == 'yandex':
                params = {
                    'text': query,
                    'lr': 213  # English results
                }
            else:  # SearX instances
                params = {
                    'q': query,
                    'categories': 'general',
                    'language': 'en'
                }
            
            logger.info(f"Making request to {engine_config['name']} with query: {query}")
            response = self._make_request(engine_config['url'], params)
            
            if response and response.status_code == 200:
                logger.info(f"Got response from {engine_config['name']}, parsing...")
                results = engine_config['parser'](response.text)
                logger.info(f"Parsed {len(results)} results from {engine_config['name']}")
                return results
            else:
                logger.warning(f"No valid response from {engine_config['name']}")
        
        except Exception as e:
            logger.error(f"Error searching with {engine_config['name']}: {e}")
        
        return []
    
    def search(self, query: str) -> List[Dict]:
        """
        Main search method called by the AI assistant
        
        Args:
            query (str): The search query
            
        Returns:
            List[Dict]: List of search results with title, url, and snippet
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        query = query.strip()
        logger.info(f"Searching for: {query}")
        
        all_results = []
        
        # Try each search engine until we get good results
        for engine in self.search_engines:
            logger.info(f"Trying search engine: {engine['name']}")
            
            results = self._search_with_engine(query, engine)
            
            if results:
                logger.info(f"Successfully found {len(results)} results from {engine['name']}")
                all_results.extend(results)
                
                # If we have enough results, stop searching
                if len(all_results) >= 6:
                    break
            else:
                logger.warning(f"No results from {engine['name']}, trying next engine...")
        
        # Remove duplicates and limit to 6 results
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            if result.url not in seen_urls and len(unique_results) < 6:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Convert to dictionary format for easy consumption
        formatted_results = []
        for result in unique_results:
            formatted_results.append({
                'title': result.title,
                'url': result.url,
                'snippet': result.snippet,
                'source': result.source
            })
        
        logger.info(f"Returning {len(formatted_results)} unique results")
        return formatted_results
    
    def search_and_summarize(self, query: str) -> str:
        """
        Search and return results in a formatted string suitable for AI consumption
        
        Args:
            query (str): The search query
            
        Returns:
            str: Formatted search results as a string
        """
        results = self.search(query)
        
        if not results:
            return f"No search results found for query: {query}"
        
        summary = f"Search results for '{query}':\n\n"
        
        for i, result in enumerate(results, 1):
            summary += f"{i}. **{result['title']}**\n"
            summary += f"   URL: {result['url']}\n"
            if result['snippet']:
                summary += f"   Summary: {result['snippet']}\n"
            summary += f"   Source: {result['source']}\n\n"
        
        return summary

# Convenience functions for direct usage
def web_search(query: str) -> List[Dict]:
    """
    Simple function to perform a web search
    
    Args:
        query (str): Search query
        
    Returns:
        List[Dict]: Search results
    """
    agent = WebSearchAgent()
    return agent.search(query)

def web_search_summary(query: str) -> str:
    """
    Simple function to get formatted search results
    
    Args:
        query (str): Search query
        
    Returns:
        str: Formatted search results
    """
    agent = WebSearchAgent()
    return agent.search_and_summarize(query)

# Example usage and testing
if __name__ == "__main__":
    # Initialize the search agent
    search_agent = WebSearchAgent()
    
    # Example search
    test_query = "Latest Minecraft updates"
    
    #print(f"Testing search for: {test_query}")
    #print("=" * 50)
    
    # Method 1: Get structured results
    results = search_agent.search(test_query)
    #print(f"Found {len(results)} results:")
    # for i, result in enumerate(results, 1):

        #print(f"\n{i}. {result['title']}")
        #print(f"   URL: {result['url']}")
        # print(f"   Snippet: {result['snippet'][:100]}...")
        # print(f"   Snippet: {result['snippet']}")
    
    #print("\n" + "=" * 50)
    
    # Method 2: Get formatted summary
    if results:
        summary = search_agent.search_and_summarize(test_query)
        print("Formatted summary:")
        print(summary)
        #print(summary[:500] + "..." if len(summary) > 500 else summary)