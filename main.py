import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import time
import os
import gc

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Book Recommendation Service",
    description="Combined API and Dashboard for book recommendations",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for lazy loading
_books = None
_db_books = None
_embedding = None

def get_books():
    global _books
    if _books is None:
        # Load only necessary columns
        _books = pd.read_csv("books_with_emotions.csv", usecols=[
            'isbn13', 'title', 'authors', 'description', 'thumbnail', 
            'simple_categories', 'joy', 'surprise', 'anger', 'fear', 'sadness'
        ])
        _books["large_thumbnail"] = _books["thumbnail"] + "&fife=w800"
        _books["large_thumbnail"] = np.where(
            _books["large_thumbnail"].isna(), 
            "sample-cover.png", 
            _books["large_thumbnail"]
        )
    return _books

def get_embedding():
    global _embedding
    if _embedding is None:
        _embedding = HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2')
    return _embedding

def get_db():
    global _db_books
    if _db_books is None:
        # Load documents in chunks
        raw_documents = TextLoader("tagged_descriptions.txt").load()
        text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
        documents = text_splitter.split_documents(raw_documents)
        
        # Create Chroma DB with persistence
        persist_directory = "chroma_db"
        if not os.path.exists(persist_directory):
            _db_books = Chroma.from_documents(
                documents, 
                embedding=get_embedding(),
                persist_directory=persist_directory
            )
            _db_books.persist()
        else:
            _db_books = Chroma(
                persist_directory=persist_directory,
                embedding_function=get_embedding()
            )
    return _db_books

# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    query: str
    category: Optional[str] = "All"
    tone: Optional[str] = "All"
    initial_top_k: Optional[int] = 50
    final_top_k: Optional[int] = 16

class BookRecommendation(BaseModel):
    title: str
    authors: str
    description: str
    thumbnail: str
    category: str
    emotions: dict

class RecommendationResponse(BaseModel):
    recommendations: List[BookRecommendation]

def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    # Add artificial delay for loading animation
    time.sleep(0.5)
    
    db = get_db()
    books = get_books()
    
    recs = db.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by='joy', ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by='surprise', ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by='anger', ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by='fear', ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by='sadness', ascending=False, inplace=True)

    return book_recs

def recommend_books(query: str, category: str, tone: str):
    if not query.strip():
        return []
        
    recommendations = retrieve_semantic_recommendations(query=query, category=category, tone=tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")

        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # Enhanced caption with HTML formatting
        caption = f"""
        <div class='book-card'>
            <h3 style='margin: 0; color: #2c3e50;'>{row['title']}</h3>
            <p style='color: #7f8c8d; margin: 5px 0;'>by {authors_str}</p>
            <p style='color: #34495e; font-size: 0.9em;'>{truncated_description}</p>
        </div>
        """
        results.append((row["large_thumbnail"], caption))

    return results

# FastAPI Routes
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Book Recommendation Service",
        "version": "1.0.0",
        "endpoints": {
            "/dashboard": "Gradio Interface",
            "/docs": "API Documentation",
            "/api/recommend": "POST - Get book recommendations",
            "/api/categories": "GET - Get available categories",
            "/api/tones": "GET - Get available emotional tones"
        }
    }

@app.get("/api/categories")
async def get_categories():
    books = get_books()
    categories = ["All"] + sorted(books["simple_categories"].unique().tolist())
    return {"categories": categories}

@app.get("/api/tones")
async def get_tones():
    tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    return {"tones": tones}

@app.post("/api/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        recommendations = retrieve_semantic_recommendations(
            query=request.query,
            category=request.category,
            tone=request.tone,
            initial_top_k=request.initial_top_k,
            final_top_k=request.final_top_k
        )

        results = []
        for _, row in recommendations.iterrows():
            book = BookRecommendation(
                title=row["title"],
                authors=row["authors"],
                description=row["description"],
                thumbnail=row["large_thumbnail"],
                category=row["simple_categories"],
                emotions={
                    "joy": float(row["joy"]),
                    "surprise": float(row["surprise"]),
                    "anger": float(row["anger"]),
                    "fear": float(row["fear"]),
                    "sadness": float(row["sadness"])
                }
            )
            results.append(book)

        return RecommendationResponse(recommendations=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create Gradio interface
def get_categories_for_ui():
    books = get_books()
    return ["All"] + sorted(books["simple_categories"].unique())

categories = get_categories_for_ui()
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Custom CSS for animations and styling
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}

.animate-fade-in {
    animation: fadeIn 0.5s ease-in;
}

.animate-slide-up {
    animation: slideUp 0.5s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideUp {
    from { transform: translateY(20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

.book-card {
    border-radius: 12px;
    padding: 15px;
    transition: transform 0.3s ease;
}

.book-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.search-box {
    border-radius: 8px;
    border: 2px solid #e0e0e0;
    transition: border-color 0.3s ease;
}

.search-box:focus {
    border-color: #4a90e2;
}

.submit-button {
    background: linear-gradient(45deg, #4a90e2, #5cb3ff);
    border: none;
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    transition: transform 0.2s ease;
}

.submit-button:hover {
    transform: scale(1.05);
}

.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-radius: 50%;
    border-top: 3px solid #4a90e2;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
"""

with gr.Blocks(theme=gr.themes.Glass(), css=custom_css) as dashboard:
    gr.Markdown("""
    # 📚 Semantic Book Recommender
    Discover your next favorite book using AI-powered recommendations!
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            user_query = gr.Textbox(
                label="What kind of book are you looking for?",
                placeholder="e.g., A story about forgiveness and redemption...",
                elem_classes=["search-box"]
            )
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                label="Category",
                value="All",
                choices=categories,
                elem_classes=["search-box"]
            )
        with gr.Column(scale=1):
            tone_dropdown = gr.Dropdown(
                label="Emotional Tone",
                value="All",
                choices=tones,
                elem_classes=["search-box"]
            )
    
    with gr.Row():
        submit_button = gr.Button(
            "🔍 Find Recommendations",
            elem_classes=["submit-button"]
        )
    
    with gr.Row():
        loading = gr.HTML(
            value="<div class='loading' style='display: none;'></div>",
            elem_classes=["animate-fade-in"]
        )
    
    gr.Markdown("## 📖 Recommended Books")
    output = gr.Gallery(
        label="",
        columns=4,
        rows=4,
        elem_classes=["animate-slide-up"],
        show_label=False
    )
    
    def show_loading():
        return "<div class='loading'></div>"
    
    def hide_loading():
        return "<div class='loading' style='display: none;'></div>"
    
    submit_button.click(
        fn=show_loading,
        outputs=loading
    ).then(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    ).then(
        fn=hide_loading,
        outputs=loading
    )

# Mount the Gradio interface
app = gr.mount_gradio_app(app, dashboard, path="/dashboard")

if __name__ == "__main__":
    # Force garbage collection before starting
    gc.collect()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 