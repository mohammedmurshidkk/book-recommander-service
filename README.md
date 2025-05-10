# Book Recommendation Service

A sophisticated book recommendation system that uses semantic search, emotional analysis, and category-based filtering to provide personalized book recommendations.

## Features

- 🔍 Semantic search using modern NLP techniques
- 😊 Emotional tone analysis (Happy, Surprising, Angry, Suspenseful, Sad)
- 📚 Category-based filtering
- 🖼️ Rich book display with covers and descriptions
- 🎯 Two-stage recommendation process for better results
- 💻 Modern web interface using Gradio
- 🌐 RESTful API for integration with any frontend

## Project Structure

```
.
├── dashboard.py              # Gradio web interface
├── api.py                   # FastAPI service
├── data-exploration.ipynb    # Initial data analysis
├── text-classification.ipynb # Text classification implementation
├── sentiment-analysis.ipynb  # Emotional analysis of book descriptions
├── vector-search.ipynb      # Vector-based search implementation
├── books_cleaned.csv        # Base dataset of books
├── books_with_categories.csv # Books with category information
├── books_with_emotions.csv  # Books with emotional analysis
├── tagged_descriptions.txt  # Processed book descriptions
└── sample-cover.png         # Default book cover image
```

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Node.js and npm (for React frontend)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd book-recommander-service
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Gradio Interface

```bash
python dashboard.py
```

### Running the API Service

```bash
python api.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the API is running, visit `http://localhost:8000/docs` for interactive API documentation.

#### Available Endpoints

1. **GET /** - API Information
   - Returns basic API information and available endpoints

2. **GET /categories** - Get Categories
   - Returns list of available book categories
   - Response: `{"categories": ["All", "Fiction", "Non-Fiction", ...]}`

3. **GET /tones** - Get Emotional Tones
   - Returns list of available emotional tones
   - Response: `{"tones": ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]}`

4. **POST /recommend** - Get Book Recommendations
   - Request Body:
     ```json
     {
       "query": "string",
       "category": "string (optional)",
       "tone": "string (optional)",
       "initial_top_k": "integer (optional)",
       "final_top_k": "integer (optional)"
     }
     ```
   - Response:
     ```json
     {
       "recommendations": [
         {
           "title": "string",
           "authors": "string",
           "description": "string",
           "thumbnail": "string",
           "category": "string",
           "emotions": {
             "joy": "float",
             "surprise": "float",
             "anger": "float",
             "fear": "float",
             "sadness": "float"
           }
         }
       ]
     }
     ```

## Deployment

### Backend Deployment Options

1. **Render**
   - Free tier available
   - Easy deployment with GitHub integration
   - Automatic HTTPS

2. **Railway**
   - Free tier available
   - Simple deployment process
   - Good for small to medium applications

3. **Heroku**
   - Free tier available
   - Extensive documentation
   - Easy scaling options

### Frontend Deployment Options

1. **Vercel**
   - Free tier available
   - Excellent for React applications
   - Automatic deployments

2. **Netlify**
   - Free tier available
   - Great for static sites
   - Easy CI/CD integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]

## Acknowledgments

- HuggingFace for NLP models
- LangChain for document processing
- Gradio for the web interface
- Chroma for vector storage
- FastAPI for the API framework

## Author

MOHAMMED MURSHID KK
- 💼 LinkedIn: [https://www.linkedin.com/in/murshidkk/]
- 🌐 Website: [https://murshidkk.info/]