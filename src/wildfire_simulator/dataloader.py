import os
import numpy as np
import rioxarray
from dotenv import load_dotenv

class DataLoader:
    """Loads landscape GeoTIFF (path from LANDSCAPE env variable) and makes
    elevation, slope, and aspect available as 2‑D numpy arrays."""

    def __init__(self):
        # Load environment from .env file
        load_dotenv()

        tif_path = os.getenv("LANDSCAPE")
        if not tif_path:
            raise EnvironmentError(
                "Environment variable 'LANDSCAPE' is not set. "
                "Make sure your .env file contains LANDSCAPE=<path>"
            )

        # Open the GeoTIFF with rioxarray – returns an xarray.DataArray
        da = rioxarray.open_rasterio(tif_path)

        # Store the full DataArray for potential later use
        self._landscape = da

        # The first band (index 0) is elevation
        self.elevation = da.isel(band=0).values

        # The second band (index 1) is slope
        self.slope = da.isel(band=1).values

        # The third band (index 2) is aspect
        self.aspect = da.isel(band=2).values

        # Band 3: fuel model (FBFM40)
        self.fuel = da.isel(band=3).values

        # Band 4: canopy cover (CC)
        self.canopy_cover = da.isel(band=4).values

        # Band 5: stand height (CH)
        self.stand_height = da.isel(band=5).values

        # Band 6: canopy base height (CBH)
        self.canopy_base_height = da.isel(band=6).values

        # Band 7: canopy bulk density (CBD)
        self.canopy_bulk_density = da.isel(band=7).values
