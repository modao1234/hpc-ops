import pytest

from benchmark.iHC.bench_ihc import compare


@pytest.mark.parametrize("invalid_us", [None, float("nan"), float("inf"), 0.0, -1.0])
@pytest.mark.parametrize("failed_side", ["baseline", "head"])
def test_benchmark_check_rejects_failed_timing(invalid_us, failed_side):
    baseline = {"passing": 100.0, "failing": invalid_us if failed_side == "baseline" else 100.0}
    results = [
        {"key": "passing", "median_us": 100.0},
        {"key": "failing", "median_us": invalid_us if failed_side == "head" else 100.0},
    ]

    assert not compare(results, baseline, threshold=0.05)


@pytest.mark.parametrize(("head_us", "passes"), [(105.0, True), (106.0, False)])
def test_benchmark_check_regression_threshold(head_us, passes):
    assert compare([{"key": "case", "median_us": head_us}], {"case": 100.0}, 0.05) == passes
