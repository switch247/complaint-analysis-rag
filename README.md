# Intelligent Complaint Analysis for Financial Services
## RAG-Powered Chatbot for CrediTrust Financial

### Project Overview
CrediTrust Financial receives thousands of customer complaints per month. This project aims to build a Retrieval-Augmented Generation (RAG) powered chatbot that transforms raw, unstructured complaint data into actionable insights. The tool empowers Product Managers, Support, and Compliance teams to ask natural language questions and receive evidence-backed answers derived directly from customer feedback.

### Business Objective
- **Decrease analysis time**: Reduce time to identify complaint trends from days to minutes.
- **Empower non-technical teams**: Enable direct access to insights without data analyst intervention.
- **Proactive resolution**: Shift from reactive to proactive problem identification.

### Data Source
The project uses complaint data from the **Consumer Financial Protection Bureau (CFPB)**, focusing on:
- Credit Cards
- Personal Loans
- Savings Accounts
- Money Transfers

### Project Structure
```
rag-complaint-chatbot/
├── data/
│   ├── raw/                # Original datasets (CFPB data)
│   └── processed/          # Cleaned and filtered data
├── docs/                   # Project documentation
├── experiments/            # Experimental notebooks
├── notebooks/              # Analysis and prototyping notebooks
├── src/                    # Source code
│   ├── analysis/           # EDA and statistical analysis
│   ├── pipeline/           # RAG pipeline components
│   ├── preprocessing/      # Data cleaning and chunking
│   └── visualisation/      # Plotting utilities
├── vector_store/           # Persisted FAISS/ChromaDB index
├── app.py                  # Gradio/Streamlit interface
├── requirements.txt        # Python dependencies
└── README.md
```

### Key Features
1.  **Data Preprocessing**: Cleaning and filtering CFPB complaint data.
2.  **Vector Search**: Semantic search using embeddings (e.g., `all-MiniLM-L6-v2`) and vector stores (ChromaDB/FAISS).
3.  **RAG Pipeline**: Retrieving relevant complaint narratives and generating answers using an LLM.
4.  **Interactive UI**: A user-friendly chat interface (Gradio/Streamlit) for querying the system.

### Setup and Installation

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd complaint-analysis-rag
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\Activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

#### 1. Data Preparation
Run the preprocessing scripts to clean data and generate embeddings.
```bash
python src/preprocessing/data_cleaning.py
# Followed by embedding generation script (if applicable)
```

#### 2. Run the Application
Launch the chat interface.
```bash
python app.py
```

### Tasks
- [x] **Task 1**: Exploratory Data Analysis and Data Preprocessing.
- [ ] **Task 2**: Text Chunking, Embedding, and Vector Store Indexing.
- [ ] **Task 3**: Building the RAG Core Logic and Evaluation.
- [ ] **Task 4**: Creating an Interactive Chat Interface.

### Technologies
- **Python**
- **LangChain** (for RAG pipeline)
- **ChromaDB / FAISS** (Vector Database)
- **Hugging Face / Sentence Transformers** (Embeddings)
- **Gradio / Streamlit** (User Interface)
- **Pandas / NumPy** (Data Manipulation)

### License
[MIT License](LICENSE)

