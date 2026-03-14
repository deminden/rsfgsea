from pathlib import Path

import pytest

import rsfgseapy


def write_gmt(tmp_path: Path) -> Path:
    gmt_path = tmp_path / "test.gmt"
    gmt_path.write_text("PW_A\tdesc\tg1\tg2\nPW_B\tdesc\tg3\tg4\n", encoding="utf-8")
    return gmt_path


def test_cpu_simple_mode_runs(tmp_path: Path) -> None:
    gmt_path = write_gmt(tmp_path)
    ranks = {"g1": 2.0, "g2": 1.0, "g3": -1.0, "g4": -2.0}

    results = rsfgseapy.run_gsea_py(
        ranks=ranks,
        gmt_path=str(gmt_path),
        mode="simple",
        nPermSimple=100,
        nperm=100,
    )

    assert len(results) == 2
    assert {row["pathway"] for row in results} == {"PW_A", "PW_B"}


def test_invalid_mode_raises_value_error(tmp_path: Path) -> None:
    gmt_path = write_gmt(tmp_path)
    ranks = {"g1": 2.0, "g2": 1.0}

    with pytest.raises(ValueError, match="Invalid mode"):
        rsfgseapy.run_gsea_py(
            ranks=ranks,
            gmt_path=str(gmt_path),
            mode="bogus",
        )


def test_gpu_requires_gpu_feature_when_not_built(tmp_path: Path) -> None:
    gmt_path = write_gmt(tmp_path)
    ranks = {"g1": 2.0, "g2": 1.0, "g3": -1.0, "g4": -2.0}

    try:
        rsfgseapy.run_gsea_py(
            ranks=ranks,
            gmt_path=str(gmt_path),
            mode="fgsea",
            gpu=True,
            nPermSimple=10,
        )
    except RuntimeError as err:
        assert "gpu" in str(err).lower()
    except Exception as err:  # pragma: no cover
        pytest.fail(f"unexpected exception type: {type(err).__name__}: {err}")


def test_write_enrichment_plot_png_py(tmp_path: Path) -> None:
    ranks = {"g1": 2.0, "g2": 1.0, "g3": -1.0, "g4": -2.0}
    output_path = tmp_path / "plot.png"

    rsfgseapy.write_enrichment_plot_png_py(
        ranks=ranks,
        pathway_genes=["g1", "g2"],
        output_path=str(output_path),
        pathway_name="PW_A",
        width_inches=1.2,
        height_inches=1.0,
        dpi=300,
    )

    assert output_path.exists()
    data = output_path.read_bytes()
    assert data.startswith(b"\x89PNG")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    assert width == 360
    assert height == 300


def test_write_gsea_table_plot_png_py(tmp_path: Path) -> None:
    ranks = {"g1": 2.0, "g2": 1.0, "g3": -1.0, "g4": -2.0}
    output_path = tmp_path / "table.png"
    results = [
        {"pathway": "PW_A", "nes": 1.5, "pval": 0.01, "padj": 0.02},
        {"pathway": "PW_B", "nes": -1.4, "pval": 0.02, "padj": 0.03},
    ]

    rsfgseapy.write_gsea_table_plot_png_py(
        ranks=ranks,
        pathways=[("PW_A", ["g1", "g2"]), ("PW_B", ["g3", "g4"])],
        results=results,
        output_path=str(output_path),
        width_inches=7.0,
        dpi=300,
    )

    assert output_path.exists()
    data = output_path.read_bytes()
    assert data.startswith(b"\x89PNG")
