import pytest

from esa.load.retry import LoadError, call_with_retry


def test_succeeds_on_third_attempt():
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("slow")
        return "ok"

    assert call_with_retry(fn, attempts=4, base_delay_s=2.0, sleep=sleeps.append) == "ok"
    assert sleeps == [2.0, 4.0]


def test_raises_load_error_after_last_attempt():
    def fn():
        raise TimeoutError("slow")

    with pytest.raises(LoadError):
        call_with_retry(fn, attempts=2, base_delay_s=0.0, sleep=lambda s: None)
