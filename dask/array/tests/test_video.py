import numpy as np
import dask
from skimage.feature import hog
from numcodecs.blosc import Blosc
# import cv2
import pims
from pims import FramesSequence
import dask.array as da
import dask_image.imread
import time

import pytest
from dask.array.image import imread


def empty_zarr_like(
    arr,
    url,
    component=None,
    storage_options=None,
    overwrite=False,
    **kwargs,
):
    """Create an empty array with the same shape as `arr`.

    See https://zarr.readthedocs.io for details about the format.

    Parameters
    ----------
    arr: dask.array
        Data to store
    url: Zarr Array or str or MutableMapping
        Location of the data. A URL can include a protocol specifier like s3://
        for remote data. Can also be any MutableMapping instance, which should
        be serializable if used in multiple processes.
    component: str or None
        If the location is a zarr group rather than an array, this is the
        subcomponent that should be created/over-written.
    storage_options: dict
        Any additional parameters for the storage backend (ignored for local
        paths)
    overwrite: bool
        If given array already exists, overwrite=False will cause an error,
        where overwrite=True will replace the existing data.
    **kwargs:
        Passed to the :func:`zarr.creation.create` function, e.g., compression options.

    Raises
    ------
    ValueError
        If ``arr`` has unknown chunk sizes, which is not supported by Zarr.

    See Also
    --------
    dask.array.Array.compute_chunk_sizes

    """
    import zarr
    from dask.array.core import _check_regular_chunks
    from fsspec import get_mapper

    if np.isnan(arr.shape).any():
        raise ValueError(
            "Saving a dask array with unknown chunk sizes is not "
            "currently supported by Zarr.%s" % unknown_chunk_message
        )

    if not _check_regular_chunks(arr.chunks):
        raise ValueError(
            "Attempt to save array to zarr with irregular "
            "chunking, please call `arr.rechunk(...)` first."
        )

    storage_options = storage_options or {}

    if isinstance(url, str):
        mapper = get_mapper(url, **storage_options)
    else:
        # assume the object passed is already a mapper
        mapper = url

    chunks = [c[0] for c in arr.chunks]

    z = zarr.create(
        shape=arr.shape,
        chunks=chunks,
        dtype=arr.dtype,
        store=mapper,
        path=component,
        overwrite=overwrite,
        **kwargs,
    )
    return z


def as_grey(frame):
    """Convert a 2D image or array of 2D images to greyscale.

    This weights the color channels according to their typical
    response to white light.

    It does nothing if the input is already greyscale.
    (Copied and slightly modified from pims source code.)
    """
    if len(frame.shape) == 2 or frame.shape[-1] != 3:  # already greyscale
        return frame
    else:
        red = frame[..., 0]
        green = frame[..., 1]
        blue = frame[..., 2]
        return 0.2125 * red + 0.7154 * green + 0.0721 * blue


def crop_frame(video_object):

    coords = cv2.selectROI("ROI Selection", video_object)
    cv2.destroyAllWindows()

    return coords

def make_hogs(frames, coords, kwargs):

    # frames will be a chunk of elements from the dask array
    # coords are the cropping coordinates used for selecting subset of image
    # kwargs are the keyword arguments that hog() expects
    # Example:
    # kwargs = dict(
    # orientations=8,
    # pixels_per_cell=(32, 32),
    # cells_per_block=(1, 1),
    # transform_sqrt=True,
    # visualize=True
    # )

    # Perform cropping procedure upon every frame, the : slice,
    # crop the x coordinates in the second slice, and crop the y
    # coordinates in the third slice. Save this new array as 
    # new_frames
    new_frames = frames[
        :,
        coords[1]:coords[1] + coords[3],
        coords[0]:coords[0] + coords[2]
    ]

    # Get the number of frames and shape for making
    # np.arrays of hog descriptors and images later
    nframes = new_frames.shape[0]
    first_frame = new_frames[0]

    # Use first frame to generate hog descriptor np.array and
    # np.array of a hog image
    hog_descriptor, hog_image = hog(
        first_frame,
        **kwargs
        )

    # Make empty numpy array that equals the number of frames passed into
    # the function, use the fed in datatype as the datatype of the images
    # and descriptors, and make the arrays shaped as each object's shape
    hog_images = np.empty((nframes,) + hog_image.shape, dtype=hog_image.dtype)
    hog_descriptors = np.empty((nframes,) + hog_descriptor.shape, dtype=hog_descriptor.dtype)

    # Until I edit the hog code, perform the hog calculation upon each
    # frame in a loop and append them to their respective np arrays
    start = time.perf_counter()
    for index, image in enumerate(new_frames):
        hog_descriptor, hog_image = hog(image, **kwargs)
        hog_descriptors[index, ...] = hog_descriptor
        hog_images[index, ...] = hog_image
    end = time.perf_counter() - start
    print(f"MADE HOGS IN {end}")
    
    return hog_descriptors, hog_images


def get_ith_tuple_element(tuple_, i=0):
    return  tuple_[i]


def normalize_hog_desc_dims(tuple_):
    # add more dimensions (each of length 1) to the hog descriptor in order
    # to match the number of dimensions of the hog_image
    descriptor = tuple_[0]
    image = tuple_[1]
    if descriptor.ndim >= image.ndim:
        return tuple_[0]
    else:
        return  np.expand_dims(
            tuple_[0], axis=list(range(descriptor.ndim, image.ndim))
        )


def _warn(message:str, category:str='', stacklevel:int=1, source:str=''): # need hints to work with pytorch
    pass # In the future, we can implement filters here. For now, just mute everything.


def test_it():
    from dask.distributed import Client
    import warnings
    
    warnings.warn = _warn
    program_start = time.perf_counter()

    video_path = r"C:\Users\heamu\lab\jupyter\test_vid.mp4"
    
    pims.ImageIOReader.class_priority = 100  # we set this very high in order to force dask's imread() to use this reader [via pims.open()]
    # client.scatter(pims.ImageIOReader)
    
    # original_video = imread(video_path, imread=pims.ImageIOReader)    
    original_video = dask_image.imread.imread(video_path, nframes=32)
    print(f'{original_video=}')

    # Turn pims frame into numpy array that opencv will take for cropping image
    # coords = crop_frame(np.array(original_video[0]))
    coords = [100, 100, 908, 506]
    
    # kwargs to use for generating both hog images and hog_descriptors
    kwargs = dict(
        orientations=8,
        pixels_per_cell=(32, 32),
        cells_per_block=(1, 1),
        transform_sqrt=True,
        visualize=True
    )
    
    grey_frames = original_video.map_blocks(as_grey, drop_axis=-1)
    
    # grey_frames = grey_frames.rechunk({0: 100})
    
    meta = np.array([[[]]])
    
    dtype = grey_frames.dtype
    
    my_hogs = grey_frames.map_blocks(
        make_hogs,
        coords=coords,
        dtype=dtype,
        meta=meta,
        kwargs=kwargs,
    )



    # my_hogs = my_hogs.persist()
    
    # first determine the output hog shapes from the first grey-scaled image so that
    # we can use them for all other images: 
    first_hog_descr, first_hog_image = make_hogs(
        grey_frames[:1, ...].compute(), coords, kwargs
    )

    hog_images = my_hogs.map_blocks(
        get_ith_tuple_element,
        i = 1,
        chunks=(grey_frames.chunks[0],) + first_hog_image.shape[1:],
        dtype=first_hog_image.dtype,
        meta=meta
    )

    descr_array_chunks = (grey_frames.chunks[0],) + first_hog_descr.shape[1:]
    if first_hog_descr.ndim <= first_hog_image.ndim:
        # we will keep the missing axes but give them each a size of 1
        new_axes = []
        n_missing_dims = first_hog_image.ndim - first_hog_descr.ndim
        descr_array_chunks += (1,)*n_missing_dims
    else:
        new_axes = list(range(first_hog_image.ndim - 1, first_hog_descr.ndim - 1))

    # Do not use `drop_axes` here!  `drop_axes` will attempt to concatenate the
    # tuples, which is undesirable.  Instead, use `squeeze()` later to drop the
    # unwanted axes.
    hog_descriptors = my_hogs.map_blocks(
        normalize_hog_desc_dims,
        new_axis=new_axes,
        chunks=descr_array_chunks,
        dtype=first_hog_descr.dtype,
        meta=meta,
    )
    hog_descriptors = hog_descriptors.squeeze(-1)  # here's where we drop the last dimension

    compressor = Blosc(cname='zstd', clevel=1)
    zarr_images = empty_zarr_like(hog_images, r"C:\Users\heamu\lab\jupyter\hog_images\data.zarr", compressor=compressor, overwrite=True)
    zarr_descriptors = empty_zarr_like(hog_descriptors, r"C:\Users\heamu\lab\jupyter\hog_descriptors\data.zarr", compressor=compressor, overwrite=True)
    with dask.config.set(scheduler='processes'):
        da.to_zarr(hog_images, zarr_images)
        da.to_zarr(hog_descriptors, zarr_descriptors)

    print("Data written to zarr! Hooray!")

    program_end = time.perf_counter() - program_start
    print(f"PROGRAM RUNTIME: {program_end}")

    # client.shutdown()