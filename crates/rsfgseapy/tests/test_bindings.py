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
