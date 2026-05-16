import numpy as np

from wildfire_simulator.forward_burn_process import ForwardBurnProcess

def test_forward_burn_process(dataset):
    burner = ForwardBurnProcess()

    frame = dataset[0]

    # burner masks fire channels 8 (mask) and 9 (arrival) / set to 0
    # all pixels with arrival > t
    new_frame = burner(frame, 30)
    assert new_frame[9].max() <= 30

    mask = new_frame[8] != 0
    assert mask.sum() > 0
    assert (frame[:, mask] == new_frame[:, mask]).all()
