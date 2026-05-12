import numpy as np
from pathlib import Path
import shutil

from wildfire_simulator.utils import save_frame

def test_save_frame():
    rng = np.random.default_rng()
    data = rng.random((13, 500, 500))
    
    shutil.rmtree('./tmp')

    # creates the folder if it doesn't exist
    save_frame(data, './tmp/random_frame.png')

    file_path = Path('./tmp/random_frame.png')
    assert file_path.is_file()

    size_bytes = file_path.stat().st_size
    assert size_bytes > 1000
