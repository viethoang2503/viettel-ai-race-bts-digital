import ast
import json
from pathlib import Path

import pytest


NOTEBOOKS = (
    Path("notebooks/colab_runner_hcm.ipynb"),
    Path("notebooks/colab_runner_bonsai.ipynb"),
)


def _source(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _load(path):
    return json.loads(path.read_text())


def _code_after_heading(notebook, heading):
    cells = notebook["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and heading in _source(cell):
            return _source(cells[index + 1])
    raise AssertionError(f"heading not found: {heading}")


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_all_non_magic_notebook_code_cells_parse(path):
    for index, cell in enumerate(_load(path)["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = _source(cell)
        if any(
            line.lstrip().startswith(("!", "%"))
            for line in source.splitlines()
        ):
            continue
        ast.parse(source, filename=f"{path}:cell-{index}")


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_experiment_matrix_reuses_one_lpips_model_and_guards_failed_scene(path):
    code = _code_after_heading(_load(path), "Bước 9")
    assert code.count("matrix_lpips_model = load_lpips_model()") == 1
    assert "lpips_model=matrix_lpips_model" in code
    assert "if scene.name not in result.chosen_config:" in code


def test_hcm_diagnosis_skips_empty_holdout_result():
    code = _code_after_heading(
        _load(NOTEBOOKS[0]),
        "Bước 8",
    )
    assert "if not worst:" in code
    assert "continue" in code


def test_hcm_uses_validated_reproducibility_packager():
    code = _code_after_heading(
        _load(NOTEBOOKS[0]),
        "Bước 11",
    )
    assert "package_reproducibility_bundle" in code
    assert "expected_scene_names" in code
    assert "shutil.make_archive" not in code


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_matrix_step_documents_resume_checkpoints(path):
    notebook = _load(path)
    markdown = "\n".join(
        _source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "checkpoint mỗi 5000 iteration" in markdown
