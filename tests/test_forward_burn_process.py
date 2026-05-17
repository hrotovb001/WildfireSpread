import numpy as np

from wildfire_simulator.forward_burn_process import ForwardBurnProcess

def test_forward_burn_process(dataset):
    burner = ForwardBurnProcess()

    tensor = dataset[0]

    # burner masks fire channels 8 (mask) and 9 (arrival) / set to 0
    # all pixels with arrival > t
    new_tensor = burner(tensor, 30)
    assert new_tensor[9].max() <= 30

    mask = new_tensor[8] != 0
    assert mask.sum() > 0
    assert (tensor[:, mask] == new_tensor[:, mask]).all()
