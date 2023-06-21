from pathlib import Path
from typing import Tuple, List
import implicit
import scipy
from data import load_user_artists, ArtistRetriever


class ImplicitRecommender:

    def __init__(
            self,
            artist_retriever: ArtistRetriever,
            implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:
        """Fit the model to the user artists matrix."""
        self.implicit_model.fit(user_artists_matrix)

    def recommend(
            self,
            user_id: int,
            user_artists_matrix: scipy.sparse.csr_matrix,
            n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Return the top n recommendations for the given user."""
        artist_ids, scores = self.implicit_model.recommend(
            user_id, user_artists_matrix[n], N=n
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores

    user_artists = load_user_artists(Path("../lastfmdata/user_artists.dat"))

    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("../lastfmdata/artists.dat"))

    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
    )

    recommender = ImplicitRecommender(artist_retriever, implicit_model)
    recommender.fit(user_artists)
    artists, scores = recommender.recommend(2, user_artists, n=5)

    # print results
    for artist, score in zip(artists, scores):
        print(f"{artist}: {score}")
