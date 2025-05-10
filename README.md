# Book Recommendation Service

A sophisticated book recommendation system that uses semantic search, emotional analysis, and category-based filtering to provide personalized book recommendations.

## Author

MOHAMMED MURSHID KK
- 💼 LinkedIn: [https://www.linkedin.com/in/murshidkk/]
- 🌐 Website: [https://murshidkk.info/]

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
├── main.py                   # Combined FastAPI and Gradio service
├── dashboard.py              # Gradio web interface (legacy)
├── api.py                    # FastAPI service (legacy)
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

### Running the Combined Service

```bash
python main.py
```

This will start both the API and the Gradio interface. You can access:
- API Documentation: `http://localhost:8000/docs`
- Gradio Interface: `http://localhost:8000/dashboard`
- API Root: `http://localhost:8000/`

### API Endpoints

1. **GET /** - Service Information
   - Returns basic service information and available endpoints

2. **GET /api/categories** - Get Categories
   - Returns list of available book categories
   - Response: `{"categories": ["All", "Fiction", "Non-Fiction", ...]}`

3. **GET /api/tones** - Get Emotional Tones
   - Returns list of available emotional tones
   - Response: `{"tones": ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]}`

4. **POST /api/recommend** - Get Book Recommendations
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

### Render Deployment

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure the service:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
   - **Environment Variables**: Add any required environment variables

The service will be available at:
- API Documentation: `https://your-app.onrender.com/docs`
- Gradio Interface: `https://your-app.onrender.com/dashboard`
- API Root: `https://your-app.onrender.com/`

### Railway Deployment

1. Create a new project on Railway
2. Connect your GitHub repository
3. Configure the service:
   - **Start Command**: `python main.py`
   - Add any required environment variables

### Frontend Deployment

For the React frontend, you can deploy to:

1. **Vercel** (Recommended)
   - Connect your GitHub repository
   - Set the build command: `npm run build`
   - Set the output directory: `build`
   - Add environment variables for API URL

2. **Netlify**
   - Connect your GitHub repository
   - Set the build command: `npm run build`
   - Set the publish directory: `build`
   - Add environment variables for API URL

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