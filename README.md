# Fake News Detector

An AI-powered fake news detector that analyzes news content and determines if it's likely real or fake, along with confidence levels and specific indicators of potential misinformation.

## Features

- Analyze news content for fake news patterns
- Advanced detection model with specific pattern recognition
- Detailed analysis of why content may be classified as fake or real
- Search and browsing of previously analyzed content
- Mobile-friendly responsive design

## Deployment to Render

This application can be deployed to Render as a web service. Follow these steps:

1. Sign up for Render (https://render.com)
2. Connect your GitHub repository
3. Create a new Web Service
4. Use the following settings:
   - Build Command: `pip install -r render-requirements.txt && python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`
   - Start Command: `gunicorn main:app`
   - Environment Variables:
     - MONGODB_URI: Your MongoDB connection string
     - SESSION_SECRET: A secure random string for Flask sessions
5. Click "Create Web Service"

Alternatively, you can use the `render.yaml` configuration file included in this repository for automatic deployment.

## Environment Variables

- `MONGODB_URI`: MongoDB connection string for storage (required for database storage)
- `SESSION_SECRET`: Secret key for Flask sessions

The application will fall back to in-memory storage if no MongoDB connection is available.

## Technology Stack

- **Backend**: Python, Flask
- **Database**: MongoDB (with fallback to in-memory storage)
- **Frontend**: HTML, CSS, JavaScript
- **NLP**: NLTK, scikit-learn
- **Deployment**: Render