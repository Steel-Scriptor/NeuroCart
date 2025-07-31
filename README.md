# NeuroCart
## An AI-powered platform leveraging machine learning and deep learning to deliver precise product insights and rankings for smarter decision-making.

### NeuroCart is an advanced AI-driven platform engineered to empower customers from websites like Walmart, Amazon, Flipkart, etc., and store personnel with precise, data-informed product selection tools through a sophisticated integration of machine learning (ML), deep learning (DL), and natural language processing (NLP) technologies. By analyzing product data and customer reviews, NeuroCart delivers actionable insights, detailed comparisons, and unbiased rankings, all orchestrated through a robust Python-based AI pipeline.

## Features

1. **Simple Input**: Enter the number of products to be added and then their respective product links to get started.
2. **Detailed Summaries**: Get clear overviews of a single product’s specs and reviews. (Applicable when a single product link is added)
3. **Side-by-Side Comparisons**: Compare two products with a detailed table. (Applicable when 2 similar products and their links are added)
4. **Smart Rankings**: Rank multiple products based on customer feedback and features. It uses a rigorously trained ML model (Applicable when 3 or more similar products and their links are added)
5. **Retail Insights**: Help e-commerce teams optimize search results and customer satisfaction based on the user reviews about that particular product.


## How the Pipeline Works

NeuroCart processes product links through a robust and fully automated pipeline. Here’s how the pipeline works:
1. **Choose Number of Links**: The system prompts you to enter 1, 2, or more than 2 product links from platforms like Walmart, Amazon, or Flipkart.
   - **Given 1 link**: Get a detailed summary of that particular product, including key specifications and reviews of all categories.
   - **Given 2 links**: Receive a comparison table, highlighting the major and minor differences between the products, and also indicate which type of user should purchase which type of product?
   - **Given more than 2 links**: Get a ranked list based on customer sentiment, with a user-sentiment score, with proper explanations and other relevant details.

2. **Enter Links**: Input the product(s) page(s) URLs.
3. **Data Collection**: The **Scraping Dog API** fetches product details (e.g., specs, features) and reviews, saving them as **JSON files**.

4. **File Conversion**:
   - JSON files are converted into **Markdown (MD) files that** contain product information, specifications, and reviews for each star rating.
   - JSON files are also processed into **CSV files** with only reviews, to be used later for sentiment analysis using the ML model, which is integrated as a tool in the AI Agent.
 
5. **PDF Generation**: MD files are transformed into **PDFs** using **PyMuPDF** for clean, shareable outputs.

6. **RAG System Setup**:
   - PDFs are chunked using the **Nomic embedder** and tokenized with an **AutoTokenizer**.
   - Chunks are stored in a **FAISS vector store** for fast searching.
   - A large context window ensures each chunk holds a full product’s info, with **top-k=10** chunks retrieved for queries.

7. **AI Agent Processing**:
   - An AI agent uses the **Gemini 1.5 Flash model** (via API) and a custom **XGBoost classifier** model alongwith **TF-IDF Vectorizer** model and a **Numeric Scaling** model (all stored as joblib files.
   - The XGBoost model, trained on a dataset of e-commerce reviews (e.g., Walmart, Amazon, Flipkart) with **TF-IDF vectorization** and **numeric scaling**, analyzes CSV files to classify review sentiments (positive, negative, neutral).
   - Outputs depend on the number of links:
     - **1 link**: RAG system + Gemini model provides a summary of that particular product with key specifications and category-wise review details.
     - **2 links**: RAG system + Gemini model generates a comparison table, highlighting the major and minor differences between the products, and also indicates which type of user should purchase which type of product.
     - **More than 2 links**: RAG system + XGBoost model + Gemini (along with TF-IDF Vectorizer and Numeric Scaler) sentiment analysis produces a ranked list with proper scores based on user(s) reviews and also gives explanations.

## Technology Stack used in this Pipeline

- **Python**: Core language for this pipeline.
- **Gemini 1.5 Flash API**: Summarizes product data and generates responses (via `google-generativeai`).
- **LangChain**: Powers the Retrieval-Augmented Generation (RAG) system for efficient data retrieval (`langchain`, `langchain-community`, `langchain-google-genai`).
- **Nomic Embedder**: Creates embeddings for text chunking (`langchain-huggingface`).
- **FAISS**: Vector store for storing and retrieving text embeddings (`faiss-cpu`).
- **AutoTokenizer**: Tokenizes text for processing (`transformers`).
- **scikit-learn**: Handles TF-IDF vectorization and numeric scaling for sentiment analysis.
- **XGBoost**: Classifies review sentiments (`xgboost`).
- **PyMuPDF**: Converts Markdown files to PDFs (`pymupdf`).
- **Scraping Dog API**: Fetches product data (`requests`, `bs4`, `trafilatura`).
- **NLTK**: Supports text preprocessing with tokenization, lemmatization, and stopword removal (`nltk`).
- **Additional Libraries**: `joblib` for model loading, `pandas` and `numpy` for data handling, `urlextract` for URL parsing, `fpdf` for PDF generation, `torch` and `bitsandbytes` for efficient model inference.

## Model Training Details

- **Dataset**: The XGBoost classifier was trained on a large dataset of e-commerce reviews from platforms like Walmart, Amazon, and Flipkart.
- **Preprocessing Models (to be used along with XGBoost Model)**:
  - **TF-IDF Vectorization**: Transforms review text into numerical features, saved as `tf-idf_vectorizer.joblib`.
  - **Numeric Scaling**: Normalizes features for consistent model input, saved as `numeric_scaler.joblib`.
- **Model**: The XGBoost classifier (`xgboost_classifier.joblib`) predicts sentiments (positive, negative, neutral) to rank products. The Accuracy of the XGBoost Model achieved till now is ** ~ 88%**

## Steps to Run the Pipeline

Follow these steps to set up and run NeuroCart:

1. **Clone the Repository**
2. Run the Pipeline (.ipynb) file named as **NeuroCart_Complete_Pipeline.ipynb**
3. All the dependencies are already available in the above pipeline only.

4. **Upload Required Files**  
   These are the 5 files which need to be uploaded to the pipeline:
   - **Gemini API Key**: For accessing Gemini 1.5 Flash (`.env`) file as GOOGLE_API_KEY.
   - **Scraping Dog API Key**: For data fetching from the product links, which would be added as a (`.env`) file as SCRAPINGDOG_API_KEY.
   - **TF-IDF Vectorizer Model**: Saved as `tf-idf_vectorizer.joblib` in the TF-IDF_Vectorizer_joblib_file folder.
   - **Numeric Scaler Model**: Saved as `numeric_scaler.joblib` in the Numeric_scaler_joblib_file_(Based on Standard Scaler) folder.
   - **XGBoost Model**: Saved as `xgboost_model.joblib` in the XGBoost_classifier_Model folder.

5. **Specify Number of Links**
   Enter 1, 2, or more than 2 when prompted.

6. **Input the Product Links**  
   Paste URLs from Walmart, Amazon, or Flipkart.

7. **View Results**  
   The system will display:
   - Summary for 1 product.
   - Comparison table for 2 products.
   - Ranking for more than 2 products.

## Demonstration Videos

Explore NeuroCart’s features through these interactive demos. Click the thumbnails to watch the videos:

1. **Single Product Summary**  
   *Shows a detailed summary of one product’s details and reviews.*
   - Content: Show entering one product link, generating JSON/MD/CSV/PDF files, and displaying
     a summary using the RAG system and Gemini model.
   - Length: 1:27 minutes.
   - [Watch on YouTube](https://youtu.be/FDF03xoReM8)

3. **Two-Product Comparison**  
   *Demonstrates the comparison table for two products.*
   - Content: Enter two links, show pipeline processing, and display a comparison table.
     a summary using the RAG system and Gemini model.
   - Length: 2 minutes.
   - [Watch on YouTube](https://youtu.be/Romb6w481MY)

5. **Multi-Product Ranking**  
   *Highlights ranking for multiple products based on user reviews.*
   - Content: Enter 3+ links, show XGBoost sentiment analysis and ranking output.
   - Length: 2:06 minutes.
   - [Watch on YouTube](https://youtu.be/SR3wc8JygjA)

## Impact
NeuroCart’s fusion of DL (LLM summarization), ML (XGBoost classification), and NLP (RAG and TF-IDF) creates a powerful tool for simplifying product selection. It reduces decision fatigue for customers, enhances satisfaction through objective insights, and provides a scalable solution to refine search optimization using customer reviews.
