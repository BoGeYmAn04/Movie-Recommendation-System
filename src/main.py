from fastapi import FastAPI,HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
import pickle

load_dotenv()
tmdb_api_key = os.getenv("TMDB_API_KEY")
tmdb_base_url = "https://api.themoviedb.org/3"
tmdb_image_base_url = "https://image.tmdb.org/t/p/w500"

if not tmdb_api_key:
    raise RuntimeError("TMDB_API_KEY is not set in the environment variables.")

app = FastAPI(title="Movie Recommendation System", description="A simple movie recommendation system using FastAPI.", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_path = os.path.dirname(os.path.abspath(__file__))
df_path = os.path.join(base_path, "df.pkl")
indices_path = os.path.join(base_path, "indices.pkl")
tfidf_matrix_path = os.path.join(base_path, "tfidf_matrix.pkl")
tfidf_path = os.path.join(base_path, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj :Any = None
tfidf_matrix : Any = None
tfidf_obj : Any = None

title_to_idx: Optional[Dict[str, int]] = None

class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    overview: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None

class TMDBMovieDetails(TMDBMovieCard):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []

class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None

class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]

def _norm_title(title: str) -> str:
    return title.strip().lower()

def make_img_url(path: Optional[str]) -> Optional[str]:
    if path:
        return f"{tmdb_image_base_url}{path}"
    return None

async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = tmdb_api_key

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502, 
            detail=f"Error while making request to TMDB: {str(e)}")
    if r.status_code != 200:
        raise HTTPException(
            status_code=502, 
            detail=f"TMDB API returned an error: {r.text}")
    return r.json()

async def tmdb_cards_from_results(
        results: List[dict],limit: int = 20
) -> List[TMDBMovieCard]:
    out : List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m.get("id")),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out    

async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data.get("id")),
        title=data.get("title") or data.get("name") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []),
    )

async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get(
        "/search/movie", 
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page
        },
    )

async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query, page=1)
    results = data.get("results", [])
    if results:
        return results[0]
    return None

def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    if isinstance(indices, dict):
        for k,v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    try:
        for k,v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception as e:
        raise RuntimeError(f"indices.pkl must be a dictionary mapping movie titles to indices. Error: {str(e)}")

def get_local_idx_by_title(title: str) -> int:
    global title_to_idx
    if title_to_idx is None:
        raise HTTPException(status_code=500, detail="Title to index mapping is not initialized.")
    key = _norm_title(title)
    if key in title_to_idx:
        return int(title_to_idx[key])
    raise HTTPException(
        status_code=404, 
        detail=f"Movie title '{title}' not found in local dataset."
    )

 