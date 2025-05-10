from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Book Recommendation API",
    description="API for semantic book recommendations with emotional analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your React app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and process data
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
# books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(), "sample-cover.png", books["large_thumbnail"])

raw_documents = TextLoader("tagged_descriptions.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

embedding = HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2')
db_books = Chroma.from_documents(documents, embedding=embedding)

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
    recs = db_books.similarity_search(query, k=initial_top_k)
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

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Book Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "/recommend": "POST - Get book recommendations",
            "/categories": "GET - Get available categories",
            "/tones": "GET - Get available emotional tones"
        }
    }

@app.get("/categories")
async def get_categories():
    categories = ["All"] + sorted(books["simple_categories"].unique().tolist())
    return {"categories": categories}

@app.get("/tones")
async def get_tones():
    tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    return {"tones": tones}

@app.post("/recommend", response_model=RecommendationResponse)
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

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 