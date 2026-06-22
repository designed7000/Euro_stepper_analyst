"""KNN player similarity model."""

import pandas as pd

from analysis.similarity import (
    build_similarity_model,
    find_similar_players,
    get_player_style_values,
)


def _stats():
    # Seven players so NearestNeighbors(n_neighbors=6) has enough samples.
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta"]
    return pd.DataFrame({
        "PLAYER_NAME": names,
        "TEAM_ABBREVIATION": ["T"] * 7,
        "USG_PCT": [0.30, 0.31, 0.20, 0.21, 0.15, 0.29, 0.18],
        "TS_PCT": [0.60, 0.59, 0.55, 0.54, 0.52, 0.61, 0.53],
        "AST_PCT": [0.30, 0.29, 0.10, 0.11, 0.08, 0.28, 0.09],
        "REB_PCT": [0.05, 0.06, 0.18, 0.17, 0.20, 0.05, 0.16],
        "PACE": [100, 101, 98, 99, 97, 100, 98],
        "3P_AR": [0.50, 0.49, 0.20, 0.22, 0.15, 0.48, 0.21],
    })


def test_build_model_returns_feature_cols():
    nn, scaler, cols = build_similarity_model(_stats())
    assert cols == ['USG_PCT', 'TS_PCT', 'AST_PCT', 'REB_PCT', 'PACE', '3P_AR']
    assert nn is not None and scaler is not None


def test_find_similar_players_ranks_closest_first():
    stats = _stats()
    nn, scaler, cols = build_similarity_model(stats)
    similar, row, err = find_similar_players("Alpha", stats, nn, scaler, cols)
    assert err is None
    # Beta and Zeta are the closest in style to Alpha.
    top_two = {p["Player"] for p in similar[:2]}
    assert {"Beta", "Zeta"} & top_two


def test_find_similar_unknown_player():
    stats = _stats()
    nn, scaler, cols = build_similarity_model(stats)
    similar, row, err = find_similar_players("Nobody", stats, nn, scaler, cols)
    assert similar is None
    assert err is not None


def test_get_player_style_values_scales_to_percent():
    row = _stats().iloc[0]
    vals = get_player_style_values(row)
    assert len(vals) == 6
    assert abs(vals[0] - 30.0) < 1e-6  # USG_PCT 0.30 -> 30.0
