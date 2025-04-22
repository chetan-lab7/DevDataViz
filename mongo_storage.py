import os
import logging
import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError

class MongoNewsStorage:
    """
    MongoDB storage for fake news detection results.
    """
    
    def __init__(self):
        """Initialize the connection to MongoDB."""
        try:
            # Get the connection string from environment variables
            mongodb_uri = os.environ.get("MONGODB_URI")
            if not mongodb_uri:
                raise ValueError("MONGODB_URI environment variable not set")
            
            # Connect to MongoDB
            self.client = MongoClient(mongodb_uri)
            
            # Test the connection
            self.client.admin.command('ping')
            
            # Set up database and collection
            self.db = self.client.fake_news_detector
            self.articles = self.db.articles
            
            logging.info("MongoDB connection established successfully")
        except ConnectionFailure as e:
            logging.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
        except PyMongoError as e:
            logging.error(f"MongoDB error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error initializing MongoDB storage: {str(e)}")
            raise
    
    def add_article(self, article):
        """
        Add an article to the MongoDB database.
        
        Args:
            article (dict): The article to add with keys:
                - id: Unique identifier
                - content: Brief content (preview)
                - full_content: Full article content
                - prediction: 'FAKE' or 'REAL'
                - confidence: Confidence score (0-100%)
                - timestamp: Date and time of analysis
        """
        try:
            if 'id' not in article:
                raise ValueError("Article must have an 'id' key")
            
            # Add a created_at field for sorting if not already present
            if 'created_at' not in article:
                article['created_at'] = datetime.datetime.now()
            
            # Insert the article document
            self.articles.insert_one(article)
            logging.debug(f"Added article with ID: {article['id']} to MongoDB")
        except PyMongoError as e:
            logging.error(f"MongoDB error while adding article: {str(e)}")
            raise
    
    def get_all_articles(self, limit=100):
        """
        Get stored articles in reverse chronological order.
        
        Args:
            limit (int): Maximum number of articles to retrieve
            
        Returns:
            list: List of article dictionaries
        """
        try:
            # Find all articles and sort by created_at timestamp in descending order
            cursor = self.articles.find().sort('created_at', -1).limit(limit)
            return list(cursor)
        except PyMongoError as e:
            logging.error(f"MongoDB error while retrieving articles: {str(e)}")
            return []
    
    def get_article_by_id(self, article_id):
        """
        Get an article by its ID.
        
        Args:
            article_id (str): The ID of the article to retrieve
            
        Returns:
            dict or None: The article dictionary if found, None otherwise
        """
        try:
            return self.articles.find_one({'id': article_id})
        except PyMongoError as e:
            logging.error(f"MongoDB error while retrieving article by ID: {str(e)}")
            return None
    
    def search_articles(self, query, limit=100):
        """
        Search for articles containing the query in content.
        
        Args:
            query (str): The search query
            limit (int): Maximum number of results to return
            
        Returns:
            list: List of matching article dictionaries
        """
        try:
            query = query.lower()
            # Use text search if it's set up, otherwise use regex
            cursor = self.articles.find({
                '$or': [
                    {'content': {'$regex': query, '$options': 'i'}},
                    {'full_content': {'$regex': query, '$options': 'i'}}
                ]
            }).sort('created_at', -1).limit(limit)
            return list(cursor)
        except PyMongoError as e:
            logging.error(f"MongoDB error while searching articles: {str(e)}")
            return []
    
    def get_timestamp(self):
        """
        Get the current timestamp as a formatted string.
        
        Returns:
            str: Formatted timestamp
        """
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")