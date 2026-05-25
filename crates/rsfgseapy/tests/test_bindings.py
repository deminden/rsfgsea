from pathlib import Path

import pytest

import rsfgseapy


def write_gmt(tmp_path: Path) -> Path:
    gmt_path = tmp_path / "test.gmt"
    gmt_path.write_text("PW_A\tdesc\tg1\tg2\nPW_B\tdesc\tg3\tg4\n", encoding="utf-8")
    return gmt_path


def write_expression(tmp_path: Path) -> Path:
    expression_path = tmp_path / "expression.tsv"
    expression_path.write_text(
        "gene\ts1\ts2\ts3\ts4\n"
        "g1\t1\t2\t3\t4\n"
        "g2\t1.1\t2.1\t3.1\t4.1\n"
        "g3\t4\t3\t2\t1\n"
        "g4\t2\t1\t2\t1\n",
        encoding="utf-8",
    )
    return expression_path


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


def test_cpu_blitz_mode_runs(tmp_path: Path) -> None:
    gmt_path = write_gmt(tmp_path)
    ranks = {"g1": 2.0, "g2": 1.0, "g3": -1.0, "g4": -2.0}

    results = rsfgseapy.run_gsea_py(
        ranks=ranks,
        gmt_path=str(gmt_path),
        mode="blitz",
        nPermSimple=64,
        minSize=1,
        maxSize=4,
        blitz_anchors=4,
        blitz_signature_cache=False,
    )

    assert len(results) == 2
    assert {row["pathway"] for row in results} == {"PW_A", "PW_B"}
    assert all(row["log2err"] is None for row in results)


def test_blitz_rejects_incompatible_options(tmp_path: Path) -> None:
    gmt_path = write_gmt(tmp_path)
    ranks = {"g1": 2.0, "g2": 1.0, "g3": -1.0, "g4": -2.0}

    with pytest.raises(ValueError, match="mode='blitz' supports only scoreType='std'"):
        rsfgseapy.run_gsea_py(
            ranks=ranks,
            gmt_path=str(gmt_path),
            mode="blitz",
            scoreType="pos",
        )


def test_invalid_mode_raises_value_error(tmp_path: Path) -> None:
    gmt_path = write_gmt(tmp_path)
    ranks = {"g1": 2.0, "g2": 1.0}

    with pytest.raises(ValueError, match="Invalid mode"):
        rsfgseapy.run_gsea_py(
            ranks=ranks,
            gmt_path=str(gmt_path),
            mode="bogus",
        )


def test_default_method_remains_classic(tmp_path: Path) -> None:
    gmt_path = write_gmt(tmp_path)
    ranks = {"g1": 2.0, "g2": 1.0, "g3": -1.0, "g4": -2.0}

    results = rsfgseapy.run_gsea_py(
        ranks=ranks,
        gmt_path=str(gmt_path),
        mode="simple",
        nPermSimple=50,
        nperm=50,
    )

    assert len(results) == 2


def test_decor_requires_cache(tmp_path: Path) -> None:
    gmt_path = write_gmt(tmp_path)
    ranks = {"g1": 2.0, "g2": 1.0, "g3": -1.0, "g4": -2.0}

    with pytest.raises(ValueError, match="requires decor_cache"):
        rsfgseapy.run_gsea_py(
            ranks=ranks,
            gmt_path=str(gmt_path),
            method="decor",
            mode="simple",
            nperm=50,
        )


def test_decor_builds_cache_from_expression(tmp_path: Path) -> None:
    gmt_path = write_gmt(tmp_path)
    expression_path = write_expression(tmp_path)
    cache_path = tmp_path / "decor-cache.tsv"
    ranks = {"g1": 2.0, "g2": 1.0, "g3": -1.0, "g4": -2.0}

    results = rsfgseapy.run_gsea_py(
        ranks=ranks,
        gmt_path=str(gmt_path),
        method="decor",
        mode="simple",
        nperm=50,
        seed=42,
        decor_cache=str(cache_path),
        decor_expression=str(expression_path),
    )

    assert cache_path.exists()
    assert len(results) == 2


def test_decor_accepts_named_presets(tmp_path: Path) -> None:
    gmt_path = write_gmt(tmp_path)
    expression_path = write_expression(tmp_path)
    ranks = {"g1": 2.0, "g2": 1.0, "g3": -1.0, "g4": -2.0}

    for preset in ["sensitive", "balanced", "specific", "strict"]:
        cache_path = tmp_path / f"decor-cache-{preset}.tsv"
        results = rsfgseapy.run_gsea_py(
            ranks=ranks,
            gmt_path=str(gmt_path),
            method="decor",
            mode="simple",
            nperm=25,
            seed=42,
            decor_cache=str(cache_path),
            decor_expression=str(expression_path),
            decor_preset=preset,
        )

        assert len(results) == 2


def test_decor_accepts_stringency_ladder(tmp_path: Path) -> None:
    gmt_path = write_gmt(tmp_path)
    expression_path = write_expression(tmp_path)
    ranks = {"g1": 2.0, "g2": 1.0, "g3": -1.0, "g4": -2.0}

    for stringency in [10.0, 50.0, 75.0, 95.0]:
        cache_path = tmp_path / f"decor-cache-stringency-{int(stringency)}.tsv"
        results = rsfgseapy.run_gsea_py(
            ranks=ranks,
            gmt_path=str(gmt_path),
            method="decor",
            mode="simple",
            nperm=25,
            seed=42,
            decor_cache=str(cache_path),
            decor_expression=str(expression_path),
            decor_stringency=stringency,
        )

        assert len(results) == 2


def test_decor_rejects_preset_and_stringency_together(tmp_path: Path) -> None:
    gmt_path = write_gmt(tmp_path)
    expression_path = write_expression(tmp_path)
    cache_path = tmp_path / "decor-cache.tsv"
    ranks = {"g1": 2.0, "g2": 1.0, "g3": -1.0, "g4": -2.0}

    with pytest.raises(ValueError, match="decor_preset or decor_stringency"):
        rsfgseapy.run_gsea_py(
            ranks=ranks,
            gmt_path=str(gmt_path),
            method="decor",
            mode="simple",
            nperm=25,
            seed=42,
            decor_cache=str(cache_path),
            decor_expression=str(expression_path),
            decor_preset="balanced",
            decor_stringency=50.0,
        )


def test_decor_does_not_expose_null_selection(tmp_path: Path) -> None:
    gmt_path = write_gmt(tmp_path)
    expression_path = write_expression(tmp_path)
    cache_path = tmp_path / "decor-cache.tsv"
    ranks = {"g1": 2.0, "g2": 1.0, "g3": -1.0, "g4": -2.0}

    with pytest.raises(TypeError, match="decor_null"):
        rsfgseapy.run_gsea_py(
            ranks=ranks,
            gmt_path=str(gmt_path),
            method="decor",
            mode="simple",
            nperm=25,
            seed=42,
            decor_cache=str(cache_path),
            decor_expression=str(expression_path),
            decor_null="profile",
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
