"""Pin the actual inherited MAG clustering rule described in the neural report.

This test belongs to the production suite: importing MAG needs its production
dependency stack, not merely NumPy and the focused neural-model dependencies.
It uses a tiny numeric example, without retrieval assets, model fitting or
optional skips. The implementation under test is deliberately unchanged.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def test_mag_clustering_uses_euclidean_distances_between_similarity_profiles():
    """The legacy parameter name is not a direct pairwise spectral-cosine gate."""
    from MS2LDA.Add_On.Spec2Vec.annotation_refined import agglomerative_clustering

    profiles = np.asarray([[0.95, 0.90], [0.92, 0.88], [0.84, 0.89]])
    actual = agglomerative_clustering(pd.DataFrame(profiles), cosine_similarity=0.9)
    expected = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.1,
        linkage="complete",
        metric="euclidean",
    ).fit_predict(1 - profiles)
    np.testing.assert_array_equal(
        actual[:, None] == actual, expected[:, None] == expected
    )
    assert actual[0] == actual[1] and actual[0] != actual[2]
    # These profiles themselves have near-identical angular directions. A cosine
    # threshold of 0.9 would merge all of them and is NOT the implemented rule.
    unit = profiles / np.linalg.norm(profiles, axis=1, keepdims=True)
    assert np.min(unit @ unit.T) > 0.9
