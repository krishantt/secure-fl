from pysnark.runtime import snark


@snark
def cube(x):
    return x * x * x


def test_snark():
    assert cube(3) == 27
