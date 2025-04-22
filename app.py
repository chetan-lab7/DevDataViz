import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from fake_news_model import FakeNewsDetector
from mongo_storage import MongoNewsStorage
import uuid

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_fallback_secret")

# Initialize fake news detector and storage
try:
    detector = FakeNewsDetector()
    
    # Try to use MongoDB storage, fall back to in-memory if not available
    try:
        news_storage = MongoNewsStorage()
        logging.info("Application initialized successfully with MongoDB storage")
    except Exception as mongo_err:
        from news_storage import NewsStorage
        news_storage = NewsStorage()
        logging.warning(f"MongoDB connection failed: {str(mongo_err)}. Using in-memory storage instead.")
except Exception as e:
    logging.error(f"Error initializing application: {str(e)}")
    # We'll let the error propagate to show the appropriate error page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Get news content from form
    news_content = request.form.get('news_content', '')
    
    if not news_content:
        flash('Please enter news content to analyze.', 'danger')
        return redirect(url_for('index'))
    
    try:
        # Generate a unique ID for this article
        article_id = str(uuid.uuid4())
        
        # Detect if the news is fake
        prediction, confidence = detector.predict(news_content)
        
        # Store the result
        news_storage.add_article({
            'id': article_id,
            'content': news_content[:200] + ('...' if len(news_content) > 200 else ''),
            'full_content': news_content,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': news_storage.get_timestamp()
        })
        
        # Store the result in session for displaying on the next page
        session['last_result'] = {
            'id': article_id,
            'content': news_content,
            'prediction': prediction,
            'confidence': confidence
        }
        
        return redirect(url_for('index', result='success'))
    
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        flash(f'An error occurred during analysis: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/history')
def history():
    # Get all stored articles
    articles = news_storage.get_all_articles()
    return render_template('history.html', articles=articles)

@app.route('/article/<article_id>')
def article_detail(article_id):
    article = news_storage.get_article_by_id(article_id)
    if article:
        return render_template('index.html', article=article)
    else:
        flash('Article not found.', 'danger')
        return redirect(url_for('history'))

@app.route('/search')
def search():
    query = request.args.get('query', '').lower()
    if not query:
        return redirect(url_for('history'))
    
    articles = news_storage.search_articles(query)
    return render_template('history.html', articles=articles, search_query=query)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
