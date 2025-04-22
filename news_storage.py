import datetime
import logging
from collections import OrderedDict

class NewsStorage:
    """
    A simple in-memory storage for fake news detection results.
    """
    
    def __init__(self):
        """Initialize the storage with an empty ordered dictionary."""
        self.articles = OrderedDict()
        logging.info("In-memory news storage initialized")
    
    def add_article(self, article):
        """
        Add an article to the storage.
        
        Args:
            article (dict): The article to add with keys:
                - id: Unique identifier
                - content: Brief content (preview)
                - full_content: Full article content
                - prediction: 'FAKE' or 'REAL'
                - confidence: Confidence score (0-100%)
                - timestamp: Date and time of analysis
        """
        if 'id' not in article:
            raise ValueError("Article must have an 'id' key")
        
        self.articles[article['id']] = article
        
        # Keep the storage size reasonable (max 100 articles)
        if len(self.articles) > 100:
            self.articles.popitem(last=False)  # Remove oldest item
        
        logging.debug(f"Added article with ID: {article['id']}")
    
    def get_all_articles(self):
        """
        Get all stored articles in reverse chronological order.
        
        Returns:
            list: List of article dictionaries
        """
        return list(reversed(list(self.articles.values())))
    
    def get_article_by_id(self, article_id):
        """
        Get an article by its ID.
        
        Args:
            article_id (str): The ID of the article to retrieve
            
        Returns:
            dict or None: The article dictionary if found, None otherwise
        """
        return self.articles.get(article_id)
    
    def search_articles(self, query):
        """
        Search for articles containing the query in content.
        
        Args:
            query (str): The search query
            
        Returns:
            list: List of matching article dictionaries
        """
        query = query.lower()
        results = []
        
        for article in self.articles.values():
            if (query in article.get('content', '').lower() or 
                query in article.get('full_content', '').lower()):
                results.append(article)
        
        return list(reversed(results))
    
    def get_timestamp(self):
        """
        Get the current timestamp as a formatted string.
        
        Returns:
            str: Formatted timestamp
        """
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
