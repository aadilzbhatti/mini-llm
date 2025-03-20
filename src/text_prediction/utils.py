import logging
import re


class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True

def sanitize_text(text):
    """Sanitizes the text while preserving meaningful dashes, numerical values with units, and hyperlink text."""
    
    # Preserve Markdown-style and HTML links by keeping only the text
    text = re.sub(r'\[([^\]]+)\]\(http\S+\)', r'\1', text)  # Handles Markdown links
    text = re.sub(r'<a\s+href=["\']http\S+["\']>(.*?)</a>', r'\1', text, flags=re.IGNORECASE)  # Handles HTML links

    # Remove standalone URLs
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)

    # Ensure we preserve dashes in hyphenated place names (e.g., Indianapolis–Carmel–Anderson)
    text = re.sub(r'(\w)\s*[-–]\s*(\w)', r'\1–\2', text)  # Normalize hyphens and remove unwanted spaces around them

    # Keep valid numerical values with units (e.g., "3.0 square miles (7.8 km2)")
    text = re.sub(r'(\d+(\.\d+)?)\s*([a-zA-Z²]+)', r'\1 \3', text)  # Ensures numbers and units stay together
    text = re.sub(r'\((\d+(\.\d+)?\s*[a-zA-Z²]+)\)', r'(\1)', text)  # Ensures parenthetical units remain intact

    # Preserve valid year ranges (e.g., 1992-2002)
    text = re.sub(r'(?<!\d)(\d{4})-(\d{4})(?!\d)', r'\1-\2', text)  # Ensure valid formatting

    # Remove unwanted characters but keep punctuation, parentheses, and percentage signs
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"()%-²]', '', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text