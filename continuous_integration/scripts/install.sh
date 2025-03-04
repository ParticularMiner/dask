set -xe

# TODO: Add cityhash back
# We don't have a conda-forge package for cityhash
# We don't include it in the conda environment.yaml, since that may
# make things harder for contributors that don't have a C++ compiler
# python -m pip install --no-deps cityhash

if [[ ${UPSTREAM_DEV} ]]; then
    mamba install -y -c arrow-nightlies "pyarrow>7.0"

    # FIXME https://github.com/mamba-org/mamba/issues/412
    # mamba uninstall --force numpy pandas fastparquet

    # TODO: Add development version of scipy once
    # https://github.com/dask/dask/issues/8682 is resolved
    conda uninstall --force numpy pandas fastparquet

    python -m pip install --no-deps --pre \
        -i https://pypi.anaconda.org/scipy-wheels-nightly/simple \
        numpy \
        pandas

    python -m pip install \
        --upgrade \
        locket \
        git+https://github.com/pydata/sparse \
        git+https://github.com/dask/s3fs \
        git+https://github.com/intake/filesystem_spec \
        git+https://github.com/dask/partd \
        git+https://github.com/dask/zict \
        git+https://github.com/dask/distributed \
        git+https://github.com/dask/fastparquet \
        git+https://github.com/zarr-developers/zarr-python
fi

# Install dask
python -m pip install --quiet --no-deps -e .[complete]
echo mamba list
mamba list

# For debugging
echo -e "--\n--Conda Environment (re-create this with \`conda env create --name <name> -f <output_file>\`)\n--"
mamba env export | grep -E -v '^prefix:.*$'

set +xe
