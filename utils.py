from bs4 import BeautifulSoup
from typing import Dict, Sequence
from custom_types import Podcast
from collections import OrderedDict


def parse_podcats(soup_main_podcast: BeautifulSoup) -> Sequence[Podcast]:

    soup_grid = soup_main_podcast.find("div", class_="grid grid-main")
    soup_podcasts = soup_grid.find_all("div", class_="grid-item main-grid-item")
    
    podcasts = []
    for soup_podcast in soup_podcasts:
        
        guest = soup_podcast.find("div", class_="vid-person").text.strip()
        title = soup_podcast.find("div", class_="vid-title").text.strip()
        soup_links = soup_podcast.select('a')
        
        url_transcript = None
        possible_transcript = [link for link in soup_links if "transcript" in link.text.lower()]
        if len(possible_transcript) > 0:
            url_transcript = possible_transcript[0].get('href')
    
        podcast = Podcast(guest, title, url_transcript)
        podcasts.append(podcast)

    return podcasts


def parse_transcript(soup: BeautifulSoup) -> OrderedDict:

    site_content = soup.find("div", class_="site-content")
    sections_and_headers = site_content.find_all(['div', 'h2'])
    
    content = OrderedDict()
    current_section = "default_section"
    current_name = "default_name"
    
    for element in sections_and_headers:
        
        if (element.name == 'div') and ('ts-segment' in element.get('class', [])):
            
            name = element.find("span", class_="ts-name").text.strip()
            # Same locutor after a change of section.
            if len(name) == 0:
                name = current_name
            else:
                current_name = name
            intervention = element.find("span", class_="ts-text").text.strip()
            content[current_section].append(f"{name}: {intervention}")
            
        elif element.name == 'h2':
            
            header = element.text.strip()
            if header.lower().startswith("table of content"):
                continue
            section = f"Section: {header}"
            content[section] = []
            current_section = section

    return content