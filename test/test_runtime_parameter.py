from lib.parametrics.runtime import RuntimeParameter


def test_runtime_parameter_clips_on_init():
    p = RuntimeParameter(name="a", value=10, lower=0, upper=5)
    assert p.value == 5


def test_runtime_parameter_set_value_and_bounds():
    p = RuntimeParameter(name="a", value=1, lower=0, upper=10)

    p.set(value=20)
    assert p.value == 10

    p.set(value=-5)
    assert p.value == 0

    p.set(lower=5)
    assert p.value == 5

    p.set(upper=-3)
    assert p.value == 5
    assert p.upper == 5


def test_runtime_parameter_set_vary_and_expr():
    p = RuntimeParameter(name="a", value=1)

    p.set(vary=False, expr="b*2")
    assert p.vary is False
    assert p.expr == "b*2"


def test_runtime_parameter_copy_with_creates_new():
    p1 = RuntimeParameter(name="a", value=2, lower=0, upper=10)
    p2 = p1.copy_with(value=5)

    assert p1 is not p2
    assert p2.value == 5
    assert p1.value == 2


def test_runtime_parameter_copy_with_to_existing():
    p1 = RuntimeParameter(name="a", value=2)
    p2 = RuntimeParameter(name="a", value=0)

    out = p1.copy_with(p2, value=7)

    assert out is None or out is p2
    assert p2.value == 7
    assert p1.value == 2
