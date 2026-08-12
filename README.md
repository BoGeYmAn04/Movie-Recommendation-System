# 🎬 Movie Recommendation System

A full-stack movie recommendation app that combines **content-based filtering (TF-IDF + cosine similarity)** on a local dataset with **live data from The Movie Database (TMDB) API**. Search for a movie, view its details, and get two flavors of recommendations: similar movies by plot/content, and similar movies by genre.

Built with **FastAPI** (backend) and **Streamlit** (frontend).

---

## ✨ Features

- 🔍 **Search** — autocomplete-style movie search powered by the TMDB API
- 📄 **Movie details** — poster, backdrop, overview, genres, release date
- 🧠 **Content-based recommendations** — TF-IDF vectorization + cosine similarity over a local movie dataset
- 🎭 **Genre-based recommendations** — live genre discovery via TMDB's `/discover` endpoint
- 🏠 **Home feed** — trending, popular, top rated, now playing, and upcoming movies
- ⚡ **FastAPI backend** with typed Pydantic response models
- 🖥️ **Streamlit frontend** with a simple client-side router (home / details views)

---

## 🏗️ Architecture

```
┌─────────────────┐        HTTP        ┌──────────────────┐        HTTP        ┌─────────────┐
│  Streamlit App   │  ───────────────►  │   FastAPI Server  │  ───────────────►  │  TMDB API   │
│  (frontend)       │  ◄───────────────  │   (backend)        │  ◄───────────────  │             │
└─────────────────┘                     └──────────────────┘                     └─────────────┘
                                                  │
                                                  ▼
                                        ┌──────────────────────┐
                                        │ Local TF-IDF dataset   │
                                        │ (df.pkl, indices.pkl,  │
                                        │  tfidf.pkl,             │
                                        │  tfidf_matrix.pkl)      │
                                        └──────────────────────┘
```

- The **backend** (`src/main.py`) serves movie data and recommendations, blending TMDB's live catalog with a locally trained TF-IDF model for content-based similarity.
- The **frontend** (Streamlit app) calls the backend's REST endpoints and renders poster grids, a details page, and recommendation sections.
- The **Notebook/** directory contains the exploratory data analysis and model-building work (TF-IDF vectorization, cosine similarity) used to generate the `.pkl` artifacts consumed by the backend.

---

## 📁 Project Structure

```
Movie-Recommendation-System/
├── Data/                # Raw / processed dataset(s) used to train the recommender
├── Notebook/             # Jupyter notebook(s) — EDA, preprocessing, TF-IDF model building
├── src/                  # Application source code
│   ├── main.py           # FastAPI backend (API routes, TMDB integration, TF-IDF recs)
│   ├── app.py             # Streamlit frontend
│   ├── df.pkl              # Preprocessed movie DataFrame
│   ├── indices.pkl         # Title → row-index mapping
│   ├── tfidf.pkl            # Fitted TF-IDF vectorizer
│   └── tfidf_matrix.pkl     # Precomputed TF-IDF matrix
├── .gitignore
└── README.md
```

> Note: adjust the file list above to match what's actually inside `src/` in this repo — this reflects the backend code as of the last update.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or 3.12 (pandas doesn't yet ship prebuilt wheels for the newest Python releases, so avoid 3.13+/3.14 to prevent slow source builds)
- A free [TMDB API key](https://www.themoviedb.org/settings/api)

### 1. Clone the repository

```bash
git clone https://github.com/BoGeYmAn04/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If a `requirements.txt` isn't present yet, install directly:
> ```bash
> pip install fastapi uvicorn httpx pydantic python-dotenv pandas numpy scipy streamlit requests
> ```

### 4. Configure environment variables

Create a `.env` file inside `src/` (same folder as `main.py`):

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

### 5. Run the backend

```bash
cd src
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

### 6. Run the frontend

In a separate terminal:

```bash
cd src
streamlit run app.py
```

By default the frontend points at a deployed backend URL — update the `API_BASE` variable at the top of the Streamlit file to `http://127.0.0.1:8000` if you're running the backend locally.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/home` | Home feed — `category` (`popular`, `trending`, `top_rated`, `now_playing`, `upcoming`) + `limit` |
| `GET` | `/tmdb/search` | Raw TMDB movie search by `query` |
| `GET` | `/movie/id/{tmdb_id}` | Full movie details by TMDB ID |
| `GET` | `/recommend/genre` | Genre-based recommendations for a given `tmdb_id` |
| `GET` | `/recommend/tfidf` | Content-based (TF-IDF) recommendations by local `title` |
| `GET` | `/movie/search` | Combined bundle — details + TF-IDF recs + genre recs for a text `query` |

Full interactive documentation is auto-generated by FastAPI at `/docs`.

---

## 🧠 How the recommendations work

- **Content-based (TF-IDF):** Each movie in the local dataset is represented as a TF-IDF vector (typically built over combined metadata like overview, genres, keywords, and/or cast). Recommendations are produced by computing cosine similarity between a query movie's vector and every other vector in the matrix, then returning the top-N closest matches.
- **Genre-based:** Uses TMDB's `/discover/movie` endpoint filtered by the queried movie's primary genre, sorted by popularity — this works for any movie in TMDB's live catalog, not just ones in the local dataset.

---

## 🛠️ Tech Stack

- **Backend:** FastAPI, Pydantic, httpx, pandas, NumPy, scikit-learn (for TF-IDF/cosine similarity)
- **Frontend:** Streamlit
- **Data source:** [TMDB API](https://www.themoviedb.org/documentation/api)
- **Model artifacts:** scikit-learn TF-IDF vectorizer, pickled DataFrame and similarity matrix

---

## 📦 Deployment

The backend is deployable to any ASGI-compatible host (Render, Railway, Fly.io, etc.). Notes:

- Pin your Python runtime to **3.11 or 3.12** to avoid slow/failing pandas source builds on hosts that default to newer Python versions.
- Set `TMDB_API_KEY` as an environment variable on your hosting platform's dashboard — `.env` files are typically not deployed automatically.
- Update the Streamlit app's `API_BASE` to point at your deployed backend URL.

---

## 🙏 Acknowledgments

This product uses the TMDB API but is not endorsed or certified by TMDB.

---
