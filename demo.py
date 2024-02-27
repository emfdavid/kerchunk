"""
There is some considerable startup cost for the joblib workers...
but there is clearly huge potential for parallelism here.

pip install zarr
2024-02-27T18:02:59.864Z MainProcess MainThread INFO:grib demo:Interp to dataframe took 48.93 seconds

pip install git+http://github.com/emfdavid/zarr-python@dd75eb85#
2024-02-27T18:04:20.473Z MainProcess MainThread INFO:grib demo:Interp to dataframe took 17.05 seconds
"""
import logging
import xarray as xr
import fsspec
import joblib
import dask
import os
import time

logging.basicConfig(
    format="%(asctime)s.%(msecs)03dZ %(processName)s %(threadName)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger("grib demo")


if __name__ == "__main__":

    fs = fsspec.filesystem(
        protocol="reference",
        fo="hrrr.wrfsfcf.16_hour_horizon.zarr",
        remote_protocol="gcs",
    )

    ds = xr.open_dataset(
        fs.get_mapper(""),
        engine="zarr",
        drop_variables=["heightAboveGround"],  # Drop magic broken variable
        backend_kwargs=dict(
            consolidated=False,
        ),
    )
    logger.info(ds)

    n_jobs = len(os.sched_getaffinity(0)) * 2

    logger.info("Running interp to dataframe with n_jobs=%d", n_jobs)
    tic = time.time()
    with (
        joblib.parallel_config(n_jobs=n_jobs, verbose=10, backend="loky"),
        dask.config.set(scheduler="processes"),
    ):
        df = (
            ds[["2t", "dswrf", "2r"]]
            .loc[dict(valid_time=slice("2023-09-01", "2023-09-30"))]
            .interp(dict(x=[600, 800], y=[700, 400]))
            .to_dataframe()
        )
    toc = time.time()

    logger.info("Interp to dataframe took %.2f seconds", toc - tic)
    logger.info(df.describe())
