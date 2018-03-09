from hypothesis import given
import hypothesis.strategies as st

import pandas as pd
import finance as fat

@given(st.floats(min_value=1, allow_nan=False, allow_infinity=False), 
    st.floats(min_value=1, allow_nan=False, allow_infinity=False))
def test_get_pct_error(x, y) :
    assert fat.get_pct_diff(x,y) == 100*(x-y)/y


def test_use_data_frame_if_inplace() :
    d_orig = pd.DataFrame(data=[0,1,2,3], columns=['Test']);
    
    # Use original if inplace=True
    d_copy = fat.use_data_frame_if_inplace(d_orig, True)
    assert(d_copy is d_orig)

    # Use copy if inplace=False
    d_copy = fat.use_data_frame_if_inplace(d_orig, False)
    assert(d_copy is not d_orig)

