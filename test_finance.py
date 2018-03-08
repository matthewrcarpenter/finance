from hypothesis import given
import hypothesis.strategies as st

import finance as fat

@given(st.floats(min_value=1, allow_nan=False, allow_infinity=False), 
    st.floats(min_value=1, allow_nan=False, allow_infinity=False))
def test_get_pct_error(x, y) :
    assert fat.get_pct_diff(x,y) == 100*(x-y)/y
