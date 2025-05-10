import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import gradio as gr
import time

load_dotenv()

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

# Load and process data
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(), "sample-cover.png", books["large_thumbnail"])

raw_documents = TextLoader("tagged_descriptions.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

embedding = HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L6-v2')
db_books = Chroma.from_documents(documents, embedding=embedding)

def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    # Add artificial delay for loading animation
    time.sleep(0.5)
    
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

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Create the Gradio interface
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

if __name__ == "__main__":
    dashboard.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        favicon_path="sample-cover.png"
    )