from pathlib import Path

from mace.modules.rigid_pair_tp import (
    RigidPairC1EdgeEmbedding,
)


def test_c1_embedding_constructs():
    embedding = (
        RigidPairC1EdgeEmbedding(
            max_ell=2,
            multiplicity=1,
        )
    )

    assert (
        embedding.body_irreps.dim
        == 83
    )

    assert (
        embedding.edge_irreps.dim
        > 0
    )


def test_models_source_routes_c1_frame():
    import mace.modules.models as models

    path = Path(
        models.__file__
    )

    text = path.read_text()

    assert (
        "RigidPairC1EdgeEmbedding"
        in text
    )

    assert (
        'self.rigid_pair_mode == "c1_frame"'
        in text
    )

    # Construction branch plus forward routing should produce
    # multiple c1_frame references.
    assert (
        text.count(
            '"c1_frame"'
        )
        >= 3
    )
