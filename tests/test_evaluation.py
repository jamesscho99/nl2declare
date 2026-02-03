import csv
from pathlib import Path

import pytest

from text2declare.evaluation import evaluate


@pytest.fixture()
def toy_csvs(tmp_path: Path):
    gt = tmp_path / 'gt.csv'
    pred = tmp_path / 'pred.csv'
    with gt.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sentence', 'constraint'])
        w.writerow(['A before B', 'Precedence(A, B)'])
        w.writerow(['Must start with A', 'Init(A)'])
    with pred.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sentence', 'constraint'])
        w.writerow(['A before B', 'Precedence(A, B)'])
        w.writerow(['Must start with A', 'Init(A)'])
    return gt, pred


def test_evaluate_perfect(toy_csvs):
    gt, pred = toy_csvs
    precision, recall, f1, num, templ_errs = evaluate(str(gt), str(pred), alpha=2.0)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)
    assert num == 2
    assert templ_errs == 0
