"""Unit tests for callback helpers with stable, non-Dash contracts."""

import base64
import gzip
import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytest

from App.callbacks import common, network, rankings_details, run_and_load, screening
from MS2LDA.Mass2Motif import Mass2Motif


def test_calculate_motif_shares_mixed() -> None:
    spectrum = {
        "mz": [110.0],
        "intensities": [1.0],
        "metadata": {"id": "s1", "precursor_mz": 200.0},
    }
    lda = {
        "theta": {"s1": {"motif_1": 0.5, "motif_2": 0.5}},
        "beta": {
            "motif_1": {"frag@110": 0.6},
            "motif_2": {"loss@90": 0.4},
        },
    }

    shares = common.calculate_motif_shares(spectrum, lda, tolerance=0.01)

    assert shares[0] == pytest.approx({"motif_1": 0.6, "motif_2": 0.4})


def test_make_spectrum_plot_without_highlighting() -> None:
    spectrum = {"mz": [100.0], "intensities": [1.0], "metadata": {"id": "a"}}

    figure = common.make_spectrum_plot(
        spectrum,
        None,
        {},
        highlight_mode="none",
    )

    assert isinstance(figure, go.Figure)
    assert figure.data[0].marker.color == "#7f7f7f"


def test_apply_common_layout() -> None:
    figure = go.Figure()

    common.apply_common_layout(figure, ytitle="Intensity")

    assert figure.layout.bargap == 0.35
    assert figure.layout.yaxis.title.text == "Intensity"


def test_load_motifset_file_uses_the_motifdb_converter(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        common,
        "load_motifDB",
        lambda _path: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(common, "motifDB2motifs", lambda _frame: ["m1"])
    motif_file = tmp_path / "motifs.json"
    motif_file.write_text("{}", encoding="utf-8")

    assert common.load_motifset_file(str(motif_file)) == ["m1"]


def test_tab_and_upload_helpers() -> None:
    tab_content = common.toggle_tab_content("load-results-tab")
    upload_message = run_and_load.update_output("data", "x.txt")

    assert tab_content[1] == {"display": "block"}
    assert upload_message.children[-1] == "Selected file: x.txt"
    assert run_and_load.toggle_advanced_settings(1, is_open=False)
    assert not run_and_load.toggle_advanced_settings(None, is_open=False)


def test_parse_ms2lda_visualization_json_and_gzip() -> None:
    data = {"a": 1}
    raw = json.dumps(data).encode()
    encoded = base64.b64encode(raw).decode()
    plain_content = f"data:application/json;base64,{encoded}"

    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="w") as handle:
        handle.write(raw)
    encoded_gzip = base64.b64encode(compressed.getvalue()).decode()
    gzip_content = f"data:application/gzip;base64,{encoded_gzip}"

    assert run_and_load.parse_ms2lda_viz_file(plain_content) == data
    assert run_and_load.parse_ms2lda_viz_file(gzip_content) == data


def test_create_cytoscape_elements_for_fragment_and_loss() -> None:
    motif = Mass2Motif(
        frag_mz=np.asarray([100.0]),
        frag_intensities=np.asarray([0.8]),
        loss_mz=np.asarray([50.0]),
        loss_intensities=np.asarray([0.4]),
        metadata={
            "precursor_mz": 150.0,
            "losses": [{"loss_mz": 50.0, "loss_intensity": 0.4}],
        },
    )

    elements = network.create_cytoscape_elements(
        [motif],
        [],
        intensity_threshold=0.1,
    )

    identifiers = {
        element["data"]["id"] for element in elements if "id" in element.get("data", {})
    }
    assert {"motif_0", "frag_100.0", "loss_50.0"} <= identifiers


def _lda_for_rankings() -> dict:
    return {
        "beta": {"motif_a": {}, "motif_b": {}},
        "theta": {
            "doc1": {"motif_a": 0.6, "motif_b": 0.2},
            "doc2": {"motif_a": 0.5, "motif_b": 0.1},
        },
        "overlap_scores": {
            "doc1": {"motif_a": 0.2, "motif_b": 0.1},
            "doc2": {"motif_a": 0.3, "motif_b": 0.05},
        },
    }


def test_compute_motif_degrees() -> None:
    degrees = rankings_details.compute_motif_degrees(
        _lda_for_rankings(),
        0.4,
        1.0,
        0.1,
        0.3,
    )

    assert degrees[0][:2] == ("motif_a", 2)
    assert degrees[1][1] == 0


def _simple_spectra() -> list[dict]:
    return [
        {
            "mz": [150.0, 120.0],
            "intensities": [1.0, 0.5],
            "metadata": {"id": "s1", "precursor_mz": 300.0},
        },
        {
            "mz": [100.0],
            "intensities": [1.0],
            "metadata": {
                "id": "s2",
                "precursor_mz": 250.0,
                "losses": [{"loss_mz": 40.225}],
            },
        },
    ]


@pytest.mark.parametrize(
    ("query", "fragment_checked", "loss_checked", "expected_id"),
    [
        ("150.00", True, False, "s1"),
        ("40.234", False, True, "s2"),
    ],
)
def test_spectra_search_respects_channel_selection(
    query: str,
    fragment_checked: bool,
    loss_checked: bool,
    expected_id: str,
) -> None:
    rows, message = screening.update_spectra_search_table(
        _simple_spectra(),
        query,
        [0, 1000],
        fragment_checked,
        loss_checked,
        [],
    )

    assert [row["spec_id"] for row in rows] == [expected_id]
    assert message == "1 spectra pass the filter"


def test_motif_ranking_applies_massql_matches() -> None:
    rows, columns, message = rankings_details.update_motif_rankings_table(
        _lda_for_rankings(),
        [0, 1],
        [0, 1],
        "motif-rankings-tab",
        ["motif_a"],
        None,
        None,
        None,
    )

    assert [row["Motif"] for row in rows] == ["motif_a"]
    assert columns
    assert message == "2 motif(s) pass the filter, 1 displayed after MassQL query"
