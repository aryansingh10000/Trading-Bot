from decimal import Decimal as D, getcontext, ROUND_HALF_EVEN, ROUND_DOWN

getcontext().prec = 40
EIGHT = D("0.00000001")


def d(x) -> D:
    """Convert to Decimal via str() so floats don't smuggle in binary noise."""
    if isinstance(x, D):
        return x
    return D(str(x))


def q8(x) -> D:
    return d(x).quantize(EIGHT, rounding=ROUND_HALF_EVEN)


def q8_down(x) -> D:
    return d(x).quantize(EIGHT, rounding=ROUND_DOWN)


def s(x: D) -> str:
    """Plain-notation string for storage (never scientific notation)."""
    return format(d(x), "f")
