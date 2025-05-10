# Book Recommendation Service

A sophisticated book recommendation system that uses semantic search, emotional analysis, and category-based filtering to provide personalized book recommendations.

## Author

[Your Name]
- LinkedIn: [Your LinkedIn Profile]
- Email: [Your Email]
- GitHub: [Your GitHub Profile]

## Features

- 🔍 Semantic search using modern NLP techniques
- 😊 Emotional tone analysis (Happy, Surprising, Angry, Suspenseful, Sad)
- 📚 Category-based filtering
- 🖼️ Rich book display with covers and descriptions
- 🎯 Two-stage recommendation process for better results
- 💻 Modern web interface using Gradio

## Project Structure

```
.
├── dashboard.py              # Main application file
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

1. Start the application:
```bash
python dashboard.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:7860)

3. Use the interface to:
   - Enter a book description or theme
   - Select a category (optional)
   - Choose an emotional tone (optional)
   - Click "Find recommendations" to get personalized book suggestions

## How It Works

### Recommendation Process

1. **Initial Search**:
   - Takes user query and converts it to semantic embeddings
   - Performs similarity search to find top 50 matching books

2. **Filtering and Sorting**:
   - Applies category filter if specified
   - Sorts by emotional tone if selected
   - Returns top 16 most relevant results

### Technical Components

- **Semantic Search**: Uses HuggingFace's paraphrase-MiniLM-L6-v2 model for text embeddings
- **Emotional Analysis**: Implements emotion-english-distilroberta-base for sentiment analysis
- **Vector Storage**: Utilizes Chroma for efficient vector similarity search
- **Web Interface**: Built with Gradio for a modern, responsive UI

## Data Processing

The project includes several Jupyter notebooks for data processing:

- `data-exploration.ipynb`: Initial data analysis and cleaning
- `text-classification.ipynb`: Implementation of text classification
- `sentiment-analysis.ipynb`: Emotional analysis of book descriptions
- `vector-search.ipynb`: Vector-based search implementation

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

## Contact

For any questions, suggestions, or collaboration opportunities, please feel free to reach out:

- 📧 Email: [Your Email]
- 💼 LinkedIn: [Your LinkedIn Profile]
- 🌐 Website: [Your Website] (optional)
- 🐦 Twitter: [Your Twitter Handle] (optional) 