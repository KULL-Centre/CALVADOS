import pytest

from calvados.sim import _split_steps


@pytest.mark.parametrize("steps", [1, 7, 10, 11, 105])
def test_split_steps_preserves_requested_total(steps: int) -> None:
    batches = _split_steps(steps)

    assert sum(batches) == steps
    assert len(batches) <= 10
    assert all(batch > 0 for batch in batches)
