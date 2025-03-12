```python
!nvcc --version
```

    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2022 NVIDIA Corporation
    Built on Mon_Oct_24_19:12:58_PDT_2022
    Cuda compilation tools, release 12.0, V12.0.76
    Build cuda_12.0.r12.0/compiler.31968024_0



```python
!pip install --upgrade numpy pandas xgboost scikit-learn
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.2.*" "dask-cudf-cu12==25.2.*" "cuml-cu12==25.2.*" \
    "cugraph-cu12==25.2.*" "nx-cugraph-cu12==25.2.*" "cuspatial-cu12==25.2.*" \
    "cuproj-cu12==25.2.*" "cuxfilter-cu12==25.2.*" "cucim-cu12==25.2.*" \
    "pylibraft-cu12==25.2.*" "raft-dask-cu12==25.2.*" "cuvs-cu12==25.2.*" \
    "nx-cugraph-cu12==25.2.*"
```

    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (1.26.4)
    Collecting numpy
      Using cached numpy-2.2.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
    Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (2.2.3)
    Requirement already satisfied: xgboost in /usr/local/lib/python3.11/dist-packages (2.1.4)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.11/dist-packages (1.6.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/lib/python3/dist-packages (from pandas) (2022.1)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas) (2023.4)
    Requirement already satisfied: nvidia-nccl-cu12 in /usr/local/lib/python3.11/dist-packages (from xgboost) (2.25.1)
    Requirement already satisfied: scipy in /usr/local/lib/python3.11/dist-packages (from xgboost) (1.11.2)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn) (1.3.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn) (3.2.0)
    Requirement already satisfied: six>=1.5 in /usr/lib/python3/dist-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mLooking in indexes: https://download.pytorch.org/whl/cu121
    Requirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages (2.1.1+cu121)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.11/dist-packages (0.16.1+cu121)
    Requirement already satisfied: torchaudio in /usr/local/lib/python3.11/dist-packages (2.1.1+cu121)
    Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch) (3.13.1)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.11/dist-packages (from torch) (4.9.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.11/dist-packages (from torch) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch) (3.2.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch) (3.1.3)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch) (2023.6.0)
    Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.11/dist-packages (from torch) (2.1.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from torchvision) (1.26.4)
    Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from torchvision) (2.31.0)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.11/dist-packages (from torchvision) (9.5.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch) (2.1.4)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->torchvision) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/lib/python3/dist-packages (from requests->torchvision) (3.3)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->torchvision) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/lib/python3/dist-packages (from requests->torchvision) (2020.6.20)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.11/dist-packages (from sympy->torch) (1.3.0)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mLooking in indexes: https://pypi.org/simple, https://pypi.nvidia.com
    Requirement already satisfied: cudf-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (25.2.2)
    Requirement already satisfied: dask-cudf-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (25.2.2)
    Requirement already satisfied: cuml-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (25.2.1)
    Requirement already satisfied: cugraph-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (25.2.0)
    Requirement already satisfied: nx-cugraph-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (25.2.0)
    Requirement already satisfied: cuspatial-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (25.2.0)
    Requirement already satisfied: cuproj-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (25.2.0)
    Requirement already satisfied: cuxfilter-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (25.2.0)
    Requirement already satisfied: cucim-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (25.2.0)
    Requirement already satisfied: pylibraft-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (25.2.0)
    Requirement already satisfied: raft-dask-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (25.2.0)
    Requirement already satisfied: cuvs-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (25.2.1)
    Requirement already satisfied: cachetools in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (5.3.2)
    Requirement already satisfied: cuda-python<13.0a0,>=12.6.2 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (12.8.0)
    Requirement already satisfied: cupy-cuda12x>=12.0.0 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (12.2.0)
    Requirement already satisfied: fsspec>=0.6.0 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (2023.6.0)
    Requirement already satisfied: libcudf-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (25.2.2)
    Requirement already satisfied: numba-cuda<0.3.0a0,>=0.2.0 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (0.2.0)
    Requirement already satisfied: numba<0.61.0a0,>=0.59.1 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (0.60.0)
    Requirement already satisfied: numpy<3.0a0,>=1.23 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (1.26.4)
    Requirement already satisfied: nvtx>=0.2.1 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (0.2.11)
    Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (23.2)
    Requirement already satisfied: pandas<2.2.4dev0,>=2.0 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (2.2.3)
    Requirement already satisfied: pyarrow<20.0.0a0,>=14.0.0 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (15.0.0)
    Requirement already satisfied: pylibcudf-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (25.2.2)
    Requirement already satisfied: pynvjitlink-cu12 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (0.5.2)
    Requirement already satisfied: rich in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (13.9.4)
    Requirement already satisfied: rmm-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (25.2.0)
    Requirement already satisfied: typing_extensions>=4.0.0 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (4.9.0)
    Requirement already satisfied: pynvml<13.0.0a0,>=12.0.0 in /usr/local/lib/python3.11/dist-packages (from dask-cudf-cu12==25.2.*) (12.0.0)
    Requirement already satisfied: rapids-dask-dependency==25.2.* in /usr/local/lib/python3.11/dist-packages (from dask-cudf-cu12==25.2.*) (25.2.0)
    Requirement already satisfied: dask-cuda==25.2.* in /usr/local/lib/python3.11/dist-packages (from cuml-cu12==25.2.*) (25.2.0)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.11/dist-packages (from cuml-cu12==25.2.*) (1.3.2)
    Requirement already satisfied: libcuml-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (from cuml-cu12==25.2.*) (25.2.1)
    Requirement already satisfied: nvidia-cublas-cu12 in /usr/local/lib/python3.11/dist-packages (from cuml-cu12==25.2.*) (12.8.4.1)
    Requirement already satisfied: nvidia-cufft-cu12 in /usr/local/lib/python3.11/dist-packages (from cuml-cu12==25.2.*) (11.3.3.83)
    Requirement already satisfied: nvidia-curand-cu12 in /usr/local/lib/python3.11/dist-packages (from cuml-cu12==25.2.*) (10.3.9.90)
    Requirement already satisfied: nvidia-cusolver-cu12 in /usr/local/lib/python3.11/dist-packages (from cuml-cu12==25.2.*) (11.7.3.90)
    Requirement already satisfied: nvidia-cusparse-cu12 in /usr/local/lib/python3.11/dist-packages (from cuml-cu12==25.2.*) (12.5.8.93)
    Requirement already satisfied: scipy>=1.8.0 in /usr/local/lib/python3.11/dist-packages (from cuml-cu12==25.2.*) (1.11.2)
    Requirement already satisfied: treelite==4.4.1 in /usr/local/lib/python3.11/dist-packages (from cuml-cu12==25.2.*) (4.4.1)
    Requirement already satisfied: libcugraph-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (from cugraph-cu12==25.2.*) (25.2.0)
    Requirement already satisfied: pylibcugraph-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (from cugraph-cu12==25.2.*) (25.2.0)
    Requirement already satisfied: ucx-py-cu12==0.42.* in /usr/local/lib/python3.11/dist-packages (from cugraph-cu12==25.2.*) (0.42.0)
    Requirement already satisfied: networkx>=3.2 in /usr/local/lib/python3.11/dist-packages (from nx-cugraph-cu12==25.2.*) (3.2.1)
    Requirement already satisfied: geopandas>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from cuspatial-cu12==25.2.*) (1.0.1)
    Requirement already satisfied: libcuspatial-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (from cuspatial-cu12==25.2.*) (25.2.0)
    Requirement already satisfied: bokeh>=3.1 in /usr/local/lib/python3.11/dist-packages (from cuxfilter-cu12==25.2.*) (3.6.3)
    Requirement already satisfied: datashader>=0.15 in /usr/local/lib/python3.11/dist-packages (from cuxfilter-cu12==25.2.*) (0.17.0)
    Requirement already satisfied: holoviews>=1.16.0 in /usr/local/lib/python3.11/dist-packages (from cuxfilter-cu12==25.2.*) (1.20.1)
    Requirement already satisfied: jupyter-server-proxy in /usr/local/lib/python3.11/dist-packages (from cuxfilter-cu12==25.2.*) (4.4.0)
    Requirement already satisfied: panel>=1.0 in /usr/local/lib/python3.11/dist-packages (from cuxfilter-cu12==25.2.*) (1.6.1)
    Requirement already satisfied: click in /usr/local/lib/python3.11/dist-packages (from cucim-cu12==25.2.*) (8.1.7)
    Requirement already satisfied: lazy-loader>=0.4 in /usr/local/lib/python3.11/dist-packages (from cucim-cu12==25.2.*) (0.4)
    Requirement already satisfied: scikit-image<0.26.0a0,>=0.19.0 in /usr/local/lib/python3.11/dist-packages (from cucim-cu12==25.2.*) (0.21.0)
    Requirement already satisfied: libraft-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (from pylibraft-cu12==25.2.*) (25.2.0)
    Requirement already satisfied: distributed-ucxx-cu12==0.42.* in /usr/local/lib/python3.11/dist-packages (from raft-dask-cu12==25.2.*) (0.42.0)
    Requirement already satisfied: libcuvs-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (from cuvs-cu12==25.2.*) (25.2.1)
    Requirement already satisfied: zict>=2.0.0 in /usr/local/lib/python3.11/dist-packages (from dask-cuda==25.2.*->cuml-cu12==25.2.*) (3.0.0)
    Requirement already satisfied: ucxx-cu12==0.42.* in /usr/local/lib/python3.11/dist-packages (from distributed-ucxx-cu12==0.42.*->raft-dask-cu12==25.2.*) (0.42.0)
    Requirement already satisfied: libkvikio-cu12==25.2.* in /usr/local/lib/python3.11/dist-packages (from libcudf-cu12==25.2.*->cudf-cu12==25.2.*) (25.2.1)
    Requirement already satisfied: nvidia-nvcomp-cu12==4.2.0.11 in /usr/local/lib/python3.11/dist-packages (from libcudf-cu12==25.2.*->cudf-cu12==25.2.*) (4.2.0.11)
    Requirement already satisfied: dask==2024.12.1 in /usr/local/lib/python3.11/dist-packages (from rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (2024.12.1)
    Requirement already satisfied: distributed==2024.12.1 in /usr/local/lib/python3.11/dist-packages (from rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (2024.12.1)
    Requirement already satisfied: dask-expr==1.1.21 in /usr/local/lib/python3.11/dist-packages (from rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (1.1.21)
    Requirement already satisfied: libucx-cu12<1.19,>=1.15.0 in /usr/local/lib/python3.11/dist-packages (from ucx-py-cu12==0.42.*->cugraph-cu12==25.2.*) (1.18.0)
    Requirement already satisfied: cloudpickle>=3.0.0 in /usr/local/lib/python3.11/dist-packages (from dask==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (3.1.1)
    Requirement already satisfied: partd>=1.4.0 in /usr/local/lib/python3.11/dist-packages (from dask==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (1.4.2)
    Requirement already satisfied: pyyaml>=5.3.1 in /usr/lib/python3/dist-packages (from dask==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (5.4.1)
    Requirement already satisfied: toolz>=0.10.0 in /usr/local/lib/python3.11/dist-packages (from dask==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (1.0.0)
    Requirement already satisfied: importlib_metadata>=4.13.0 in /usr/local/lib/python3.11/dist-packages (from dask==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (8.6.1)
    Requirement already satisfied: jinja2>=2.10.3 in /usr/local/lib/python3.11/dist-packages (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (3.1.3)
    Requirement already satisfied: locket>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (1.0.0)
    Requirement already satisfied: msgpack>=1.0.2 in /usr/local/lib/python3.11/dist-packages (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (1.1.0)
    Requirement already satisfied: psutil>=5.8.0 in /usr/local/lib/python3.11/dist-packages (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (5.9.8)
    Requirement already satisfied: sortedcontainers>=2.0.5 in /usr/local/lib/python3.11/dist-packages (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (2.4.0)
    Requirement already satisfied: tblib>=1.6.0 in /usr/local/lib/python3.11/dist-packages (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (3.0.0)
    Requirement already satisfied: tornado>=6.2.0 in /usr/local/lib/python3.11/dist-packages (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (6.4)
    Requirement already satisfied: urllib3>=1.26.5 in /usr/local/lib/python3.11/dist-packages (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (2.0.7)
    Requirement already satisfied: libucxx-cu12==0.42.* in /usr/local/lib/python3.11/dist-packages (from ucxx-cu12==0.42.*->distributed-ucxx-cu12==0.42.*->raft-dask-cu12==25.2.*) (0.42.0)
    Requirement already satisfied: contourpy>=1.2 in /usr/local/lib/python3.11/dist-packages (from bokeh>=3.1->cuxfilter-cu12==25.2.*) (1.2.0)
    Requirement already satisfied: pillow>=7.1.0 in /usr/local/lib/python3.11/dist-packages (from bokeh>=3.1->cuxfilter-cu12==25.2.*) (9.5.0)
    Requirement already satisfied: xyzservices>=2021.09.1 in /usr/local/lib/python3.11/dist-packages (from bokeh>=3.1->cuxfilter-cu12==25.2.*) (2025.1.0)
    Requirement already satisfied: cuda-bindings~=12.8.0 in /usr/local/lib/python3.11/dist-packages (from cuda-python<13.0a0,>=12.6.2->cudf-cu12==25.2.*) (12.8.0)
    Requirement already satisfied: fastrlock>=0.5 in /usr/local/lib/python3.11/dist-packages (from cupy-cuda12x>=12.0.0->cudf-cu12==25.2.*) (0.8.2)
    Requirement already satisfied: colorcet in /usr/local/lib/python3.11/dist-packages (from datashader>=0.15->cuxfilter-cu12==25.2.*) (3.1.0)
    Requirement already satisfied: multipledispatch in /usr/local/lib/python3.11/dist-packages (from datashader>=0.15->cuxfilter-cu12==25.2.*) (1.0.0)
    Requirement already satisfied: param in /usr/local/lib/python3.11/dist-packages (from datashader>=0.15->cuxfilter-cu12==25.2.*) (2.2.0)
    Requirement already satisfied: pyct in /usr/local/lib/python3.11/dist-packages (from datashader>=0.15->cuxfilter-cu12==25.2.*) (0.5.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from datashader>=0.15->cuxfilter-cu12==25.2.*) (2.31.0)
    Requirement already satisfied: xarray in /usr/local/lib/python3.11/dist-packages (from datashader>=0.15->cuxfilter-cu12==25.2.*) (2025.1.2)
    Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /usr/local/lib/python3.11/dist-packages (from fsspec[http]>=0.6.0->cugraph-cu12==25.2.*) (3.9.1)
    Requirement already satisfied: pyogrio>=0.7.2 in /usr/local/lib/python3.11/dist-packages (from geopandas>=1.0.0->cuspatial-cu12==25.2.*) (0.10.0)
    Requirement already satisfied: pyproj>=3.3.0 in /usr/local/lib/python3.11/dist-packages (from geopandas>=1.0.0->cuspatial-cu12==25.2.*) (3.7.1)
    Requirement already satisfied: shapely>=2.0.0 in /usr/local/lib/python3.11/dist-packages (from geopandas>=1.0.0->cuspatial-cu12==25.2.*) (2.0.7)
    Requirement already satisfied: pyviz-comms>=2.1 in /usr/local/lib/python3.11/dist-packages (from holoviews>=1.16.0->cuxfilter-cu12==25.2.*) (3.0.4)
    Requirement already satisfied: llvmlite<0.44,>=0.43.0dev0 in /usr/local/lib/python3.11/dist-packages (from numba<0.61.0a0,>=0.59.1->cudf-cu12==25.2.*) (0.43.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas<2.2.4dev0,>=2.0->cudf-cu12==25.2.*) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/lib/python3/dist-packages (from pandas<2.2.4dev0,>=2.0->cudf-cu12==25.2.*) (2022.1)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas<2.2.4dev0,>=2.0->cudf-cu12==25.2.*) (2023.4)
    Requirement already satisfied: bleach in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->cuxfilter-cu12==25.2.*) (6.1.0)
    Requirement already satisfied: linkify-it-py in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->cuxfilter-cu12==25.2.*) (2.0.3)
    Requirement already satisfied: markdown in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->cuxfilter-cu12==25.2.*) (3.5.2)
    Requirement already satisfied: markdown-it-py in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->cuxfilter-cu12==25.2.*) (3.0.0)
    Requirement already satisfied: mdit-py-plugins in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->cuxfilter-cu12==25.2.*) (0.4.2)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->cuxfilter-cu12==25.2.*) (4.66.1)
    Requirement already satisfied: nvidia-ml-py<13.0.0a0,>=12.0.0 in /usr/local/lib/python3.11/dist-packages (from pynvml<13.0.0a0,>=12.0.0->dask-cudf-cu12==25.2.*) (12.570.86)
    Requirement already satisfied: imageio>=2.27 in /usr/local/lib/python3.11/dist-packages (from scikit-image<0.26.0a0,>=0.19.0->cucim-cu12==25.2.*) (2.33.1)
    Requirement already satisfied: tifffile>=2022.8.12 in /usr/local/lib/python3.11/dist-packages (from scikit-image<0.26.0a0,>=0.19.0->cucim-cu12==25.2.*) (2023.12.9)
    Requirement already satisfied: PyWavelets>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from scikit-image<0.26.0a0,>=0.19.0->cucim-cu12==25.2.*) (1.5.0)
    Requirement already satisfied: jupyter-server>=1.24.0 in /usr/local/lib/python3.11/dist-packages (from jupyter-server-proxy->cuxfilter-cu12==25.2.*) (2.12.5)
    Requirement already satisfied: simpervisor>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from jupyter-server-proxy->cuxfilter-cu12==25.2.*) (1.0.0)
    Requirement already satisfied: traitlets>=5.1.0 in /usr/local/lib/python3.11/dist-packages (from jupyter-server-proxy->cuxfilter-cu12==25.2.*) (5.14.1)
    Requirement already satisfied: nvidia-nvjitlink-cu12 in /usr/local/lib/python3.11/dist-packages (from nvidia-cufft-cu12->cuml-cu12==25.2.*) (12.8.93)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.11/dist-packages (from rich->cudf-cu12==25.2.*) (2.17.2)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=0.6.0->cugraph-cu12==25.2.*) (23.1.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=0.6.0->cugraph-cu12==25.2.*) (6.0.4)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=0.6.0->cugraph-cu12==25.2.*) (1.9.4)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=0.6.0->cugraph-cu12==25.2.*) (1.4.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.11/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=0.6.0->cugraph-cu12==25.2.*) (1.3.1)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2>=2.10.3->distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (2.1.4)
    Requirement already satisfied: anyio>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (4.2.0)
    Requirement already satisfied: argon2-cffi in /usr/local/lib/python3.11/dist-packages (from jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (23.1.0)
    Requirement already satisfied: jupyter-client>=7.4.4 in /usr/local/lib/python3.11/dist-packages (from jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (7.4.9)
    Requirement already satisfied: jupyter-core!=5.0.*,>=4.12 in /usr/local/lib/python3.11/dist-packages (from jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (5.7.1)
    Requirement already satisfied: jupyter-events>=0.9.0 in /usr/local/lib/python3.11/dist-packages (from jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (0.9.0)
    Requirement already satisfied: jupyter-server-terminals in /usr/local/lib/python3.11/dist-packages (from jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (0.5.2)
    Requirement already satisfied: nbconvert>=6.4.4 in /usr/local/lib/python3.11/dist-packages (from jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (7.14.2)
    Requirement already satisfied: nbformat>=5.3.0 in /usr/local/lib/python3.11/dist-packages (from jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (5.9.2)
    Requirement already satisfied: overrides in /usr/local/lib/python3.11/dist-packages (from jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (7.6.0)
    Requirement already satisfied: prometheus-client in /usr/local/lib/python3.11/dist-packages (from jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (0.9.0)
    Requirement already satisfied: pyzmq>=24 in /usr/local/lib/python3.11/dist-packages (from jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (24.0.1)
    Requirement already satisfied: send2trash>=1.8.2 in /usr/local/lib/python3.11/dist-packages (from jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (1.8.2)
    Requirement already satisfied: terminado>=0.8.3 in /usr/local/lib/python3.11/dist-packages (from jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (0.18.0)
    Requirement already satisfied: websocket-client in /usr/local/lib/python3.11/dist-packages (from jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (0.57.0)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.11/dist-packages (from markdown-it-py->panel>=1.0->cuxfilter-cu12==25.2.*) (0.1.2)
    Requirement already satisfied: certifi in /usr/lib/python3/dist-packages (from pyogrio>=0.7.2->geopandas>=1.0.0->cuspatial-cu12==25.2.*) (2020.6.20)
    Requirement already satisfied: six>=1.5 in /usr/lib/python3/dist-packages (from python-dateutil>=2.8.2->pandas<2.2.4dev0,>=2.0->cudf-cu12==25.2.*) (1.16.0)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.11/dist-packages (from bleach->panel>=1.0->cuxfilter-cu12==25.2.*) (0.5.1)
    Requirement already satisfied: uc-micro-py in /usr/local/lib/python3.11/dist-packages (from linkify-it-py->panel>=1.0->cuxfilter-cu12==25.2.*) (1.0.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->datashader>=0.15->cuxfilter-cu12==25.2.*) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/lib/python3/dist-packages (from requests->datashader>=0.15->cuxfilter-cu12==25.2.*) (3.3)
    Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.11/dist-packages (from anyio>=3.1.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (1.3.0)
    Requirement already satisfied: zipp>=3.20 in /usr/local/lib/python3.11/dist-packages (from importlib_metadata>=4.13.0->dask==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (3.21.0)
    Requirement already satisfied: entrypoints in /usr/local/lib/python3.11/dist-packages (from jupyter-client>=7.4.4->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (0.4)
    Requirement already satisfied: nest-asyncio>=1.5.4 in /usr/local/lib/python3.11/dist-packages (from jupyter-client>=7.4.4->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (1.6.0)
    Requirement already satisfied: platformdirs>=2.5 in /usr/local/lib/python3.11/dist-packages (from jupyter-core!=5.0.*,>=4.12->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (4.1.0)
    Requirement already satisfied: jsonschema>=4.18.0 in /usr/local/lib/python3.11/dist-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (4.21.1)
    Requirement already satisfied: python-json-logger>=2.0.4 in /usr/local/lib/python3.11/dist-packages (from jupyter-events>=0.9.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (2.0.7)
    Requirement already satisfied: referencing in /usr/local/lib/python3.11/dist-packages (from jupyter-events>=0.9.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (0.32.1)
    Requirement already satisfied: rfc3339-validator in /usr/local/lib/python3.11/dist-packages (from jupyter-events>=0.9.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (0.1.4)
    Requirement already satisfied: rfc3986-validator>=0.1.1 in /usr/local/lib/python3.11/dist-packages (from jupyter-events>=0.9.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (0.1.1)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.11/dist-packages (from nbconvert>=6.4.4->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (4.12.3)
    Requirement already satisfied: defusedxml in /usr/local/lib/python3.11/dist-packages (from nbconvert>=6.4.4->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (0.7.1)
    Requirement already satisfied: jupyterlab-pygments in /usr/local/lib/python3.11/dist-packages (from nbconvert>=6.4.4->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (0.3.0)
    Requirement already satisfied: mistune<4,>=2.0.3 in /usr/local/lib/python3.11/dist-packages (from nbconvert>=6.4.4->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (3.0.2)
    Requirement already satisfied: nbclient>=0.5.0 in /usr/local/lib/python3.11/dist-packages (from nbconvert>=6.4.4->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (0.9.0)
    Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.11/dist-packages (from nbconvert>=6.4.4->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (1.5.1)
    Requirement already satisfied: tinycss2 in /usr/local/lib/python3.11/dist-packages (from nbconvert>=6.4.4->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (1.2.1)
    Requirement already satisfied: fastjsonschema in /usr/local/lib/python3.11/dist-packages (from nbformat>=5.3.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (2.19.1)
    Requirement already satisfied: ptyprocess in /usr/local/lib/python3.11/dist-packages (from terminado>=0.8.3->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (0.7.0)
    Requirement already satisfied: argon2-cffi-bindings in /usr/local/lib/python3.11/dist-packages (from argon2-cffi->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (21.2.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=4.18.0->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (2023.12.1)
    Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.11/dist-packages (from jsonschema>=4.18.0->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (0.17.1)
    Requirement already satisfied: fqdn in /usr/local/lib/python3.11/dist-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (1.5.1)
    Requirement already satisfied: isoduration in /usr/local/lib/python3.11/dist-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (20.11.0)
    Requirement already satisfied: jsonpointer>1.13 in /usr/local/lib/python3.11/dist-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (2.4)
    Requirement already satisfied: uri-template in /usr/local/lib/python3.11/dist-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (1.3.0)
    Requirement already satisfied: webcolors>=1.11 in /usr/local/lib/python3.11/dist-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (1.13)
    Requirement already satisfied: cffi>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from argon2-cffi-bindings->argon2-cffi->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (1.16.0)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.11/dist-packages (from beautifulsoup4->nbconvert>=6.4.4->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (2.5)
    Requirement already satisfied: pycparser in /usr/local/lib/python3.11/dist-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (2.21)
    Requirement already satisfied: arrow>=0.15.0 in /usr/local/lib/python3.11/dist-packages (from isoduration->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (1.3.0)
    Requirement already satisfied: types-python-dateutil>=2.8.10 in /usr/local/lib/python3.11/dist-packages (from arrow>=0.15.0->isoduration->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (2.8.19.20240106)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m


```python
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import cudf

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

print(f"xgb {xgb.__version__}")
kfold = KFold(shuffle=True, random_state=42)
```

    PyTorch version: 2.1.1+cu121
    CUDA available: True
    CUDA version: 12.1
    GPU device count: 1
    Current device: 0
    Device name: Quadro RTX 4000
    xgb 2.1.4



```python
# fn = "train_simple.csv"
fn = "train_orig.csv"
print(f"reading {fn}")
train = pd.read_csv(f"../datasets/net/{fn}")

train = pd.concat(
    [
        train.select_dtypes("int64").astype("int32"),
        train.select_dtypes("float64").astype("float32"),
    ],
    axis=1,
)

print(f"\ntrain: {str(train.shape):>23}")
print(f"{train.columns.to_list()}")
```

    reading train.csv
    
    train:           (202033, 413)
    ['Season', 'DayNum', 'TeamID_1', 'TeamID_2', 'Margin', 'Score_pg_o_1', 'Score_poss_o_1', 'FGM_pg_o_1', 'FGM_poss_o_1', 'FGA_pg_o_1', 'FGA_poss_o_1', 'FGM3_pg_o_1', 'FGM3_poss_o_1', 'FGA3_pg_o_1', 'FGA3_poss_o_1', 'FTM_pg_o_1', 'FTM_poss_o_1', 'FTA_pg_o_1', 'FTA_poss_o_1', 'OR_pg_o_1', 'OR_poss_o_1', 'DR_pg_o_1', 'DR_poss_o_1', 'Ast_pg_o_1', 'Ast_poss_o_1', 'TO_pg_o_1', 'TO_poss_o_1', 'Stl_pg_o_1', 'Stl_poss_o_1', 'Blk_pg_o_1', 'Blk_poss_o_1', 'PF_pg_o_1', 'PF_poss_o_1', 'Poss_pg_o_1', 'Score_pg_d_1', 'Score_poss_d_1', 'FGM_pg_d_1', 'FGM_poss_d_1', 'FGA_pg_d_1', 'FGA_poss_d_1', 'FGM3_pg_d_1', 'FGM3_poss_d_1', 'FGA3_pg_d_1', 'FGA3_poss_d_1', 'FTM_pg_d_1', 'FTM_poss_d_1', 'FTA_pg_d_1', 'FTA_poss_d_1', 'OR_pg_d_1', 'OR_poss_d_1', 'DR_pg_d_1', 'DR_poss_d_1', 'Ast_pg_d_1', 'Ast_poss_d_1', 'TO_pg_d_1', 'TO_poss_d_1', 'Stl_pg_d_1', 'Stl_poss_d_1', 'Blk_pg_d_1', 'Blk_poss_d_1', 'PF_pg_d_1', 'PF_poss_d_1', 'Poss_pg_d_1', 'FGPct_o_1', 'FGPct3_o_1', 'FTPct_o_1', 'AstPct_o_1', 'AstTO_o_1', 'FGPct_d_1', 'FGPct3_d_1', 'FTPct_d_1', 'AstPct_d_1', 'AstTO_d_1', 'Score_pg_diff_1', 'Score_poss_diff_1', 'FGM_pg_diff_1', 'FGM_poss_diff_1', 'FGA_pg_diff_1', 'FGA_poss_diff_1', 'FGM3_pg_diff_1', 'FGM3_poss_diff_1', 'FGA3_pg_diff_1', 'FGA3_poss_diff_1', 'FTM_pg_diff_1', 'FTM_poss_diff_1', 'FTA_pg_diff_1', 'FTA_poss_diff_1', 'OR_pg_diff_1', 'OR_poss_diff_1', 'DR_pg_diff_1', 'DR_poss_diff_1', 'Ast_pg_diff_1', 'Ast_poss_diff_1', 'TO_pg_diff_1', 'TO_poss_diff_1', 'Stl_pg_diff_1', 'Stl_poss_diff_1', 'Blk_pg_diff_1', 'Blk_poss_diff_1', 'PF_pg_diff_1', 'PF_poss_diff_1', 'Poss_pg_diff_1', 'FGPct_diff_1', 'FGPct3_diff_1', 'FTPct_diff_1', 'AstPct_diff_1', 'AstTO_diff_1', 'sos_Score_pg_o_1', 'sos_Score_poss_o_1', 'sos_FGM_pg_o_1', 'sos_FGM_poss_o_1', 'sos_FGA_pg_o_1', 'sos_FGA_poss_o_1', 'sos_FGM3_pg_o_1', 'sos_FGM3_poss_o_1', 'sos_FGA3_pg_o_1', 'sos_FGA3_poss_o_1', 'sos_FTM_pg_o_1', 'sos_FTM_poss_o_1', 'sos_FTA_pg_o_1', 'sos_FTA_poss_o_1', 'sos_OR_pg_o_1', 'sos_OR_poss_o_1', 'sos_DR_pg_o_1', 'sos_DR_poss_o_1', 'sos_Ast_pg_o_1', 'sos_Ast_poss_o_1', 'sos_TO_pg_o_1', 'sos_TO_poss_o_1', 'sos_Stl_pg_o_1', 'sos_Stl_poss_o_1', 'sos_Blk_pg_o_1', 'sos_Blk_poss_o_1', 'sos_PF_pg_o_1', 'sos_PF_poss_o_1', 'sos_Poss_pg_o_1', 'sos_Score_pg_d_1', 'sos_Score_poss_d_1', 'sos_FGM_pg_d_1', 'sos_FGM_poss_d_1', 'sos_FGA_pg_d_1', 'sos_FGA_poss_d_1', 'sos_FGM3_pg_d_1', 'sos_FGM3_poss_d_1', 'sos_FGA3_pg_d_1', 'sos_FGA3_poss_d_1', 'sos_FTM_pg_d_1', 'sos_FTM_poss_d_1', 'sos_FTA_pg_d_1', 'sos_FTA_poss_d_1', 'sos_OR_pg_d_1', 'sos_OR_poss_d_1', 'sos_DR_pg_d_1', 'sos_DR_poss_d_1', 'sos_Ast_pg_d_1', 'sos_Ast_poss_d_1', 'sos_TO_pg_d_1', 'sos_TO_poss_d_1', 'sos_Stl_pg_d_1', 'sos_Stl_poss_d_1', 'sos_Blk_pg_d_1', 'sos_Blk_poss_d_1', 'sos_PF_pg_d_1', 'sos_PF_poss_d_1', 'sos_Poss_pg_d_1', 'sos_FGPct_o_1', 'sos_FGPct3_o_1', 'sos_FTPct_o_1', 'sos_AstPct_o_1', 'sos_AstTO_o_1', 'sos_FGPct_d_1', 'sos_FGPct3_d_1', 'sos_FTPct_d_1', 'sos_AstPct_d_1', 'sos_AstTO_d_1', 'sos_Score_pg_diff_1', 'sos_Score_poss_diff_1', 'sos_FGM_pg_diff_1', 'sos_FGM_poss_diff_1', 'sos_FGA_pg_diff_1', 'sos_FGA_poss_diff_1', 'sos_FGM3_pg_diff_1', 'sos_FGM3_poss_diff_1', 'sos_FGA3_pg_diff_1', 'sos_FGA3_poss_diff_1', 'sos_FTM_pg_diff_1', 'sos_FTM_poss_diff_1', 'sos_FTA_pg_diff_1', 'sos_FTA_poss_diff_1', 'sos_OR_pg_diff_1', 'sos_OR_poss_diff_1', 'sos_DR_pg_diff_1', 'sos_DR_poss_diff_1', 'sos_Ast_pg_diff_1', 'sos_Ast_poss_diff_1', 'sos_TO_pg_diff_1', 'sos_TO_poss_diff_1', 'sos_Stl_pg_diff_1', 'sos_Stl_poss_diff_1', 'sos_Blk_pg_diff_1', 'sos_Blk_poss_diff_1', 'sos_PF_pg_diff_1', 'sos_PF_poss_diff_1', 'sos_Poss_pg_diff_1', 'sos_FGPct_diff_1', 'sos_FGPct3_diff_1', 'sos_FTPct_diff_1', 'sos_AstPct_diff_1', 'sos_AstTO_diff_1', 'Score_pg_o_2', 'Score_poss_o_2', 'FGM_pg_o_2', 'FGM_poss_o_2', 'FGA_pg_o_2', 'FGA_poss_o_2', 'FGM3_pg_o_2', 'FGM3_poss_o_2', 'FGA3_pg_o_2', 'FGA3_poss_o_2', 'FTM_pg_o_2', 'FTM_poss_o_2', 'FTA_pg_o_2', 'FTA_poss_o_2', 'OR_pg_o_2', 'OR_poss_o_2', 'DR_pg_o_2', 'DR_poss_o_2', 'Ast_pg_o_2', 'Ast_poss_o_2', 'TO_pg_o_2', 'TO_poss_o_2', 'Stl_pg_o_2', 'Stl_poss_o_2', 'Blk_pg_o_2', 'Blk_poss_o_2', 'PF_pg_o_2', 'PF_poss_o_2', 'Poss_pg_o_2', 'Score_pg_d_2', 'Score_poss_d_2', 'FGM_pg_d_2', 'FGM_poss_d_2', 'FGA_pg_d_2', 'FGA_poss_d_2', 'FGM3_pg_d_2', 'FGM3_poss_d_2', 'FGA3_pg_d_2', 'FGA3_poss_d_2', 'FTM_pg_d_2', 'FTM_poss_d_2', 'FTA_pg_d_2', 'FTA_poss_d_2', 'OR_pg_d_2', 'OR_poss_d_2', 'DR_pg_d_2', 'DR_poss_d_2', 'Ast_pg_d_2', 'Ast_poss_d_2', 'TO_pg_d_2', 'TO_poss_d_2', 'Stl_pg_d_2', 'Stl_poss_d_2', 'Blk_pg_d_2', 'Blk_poss_d_2', 'PF_pg_d_2', 'PF_poss_d_2', 'Poss_pg_d_2', 'FGPct_o_2', 'FGPct3_o_2', 'FTPct_o_2', 'AstPct_o_2', 'AstTO_o_2', 'FGPct_d_2', 'FGPct3_d_2', 'FTPct_d_2', 'AstPct_d_2', 'AstTO_d_2', 'Score_pg_diff_2', 'Score_poss_diff_2', 'FGM_pg_diff_2', 'FGM_poss_diff_2', 'FGA_pg_diff_2', 'FGA_poss_diff_2', 'FGM3_pg_diff_2', 'FGM3_poss_diff_2', 'FGA3_pg_diff_2', 'FGA3_poss_diff_2', 'FTM_pg_diff_2', 'FTM_poss_diff_2', 'FTA_pg_diff_2', 'FTA_poss_diff_2', 'OR_pg_diff_2', 'OR_poss_diff_2', 'DR_pg_diff_2', 'DR_poss_diff_2', 'Ast_pg_diff_2', 'Ast_poss_diff_2', 'TO_pg_diff_2', 'TO_poss_diff_2', 'Stl_pg_diff_2', 'Stl_poss_diff_2', 'Blk_pg_diff_2', 'Blk_poss_diff_2', 'PF_pg_diff_2', 'PF_poss_diff_2', 'Poss_pg_diff_2', 'FGPct_diff_2', 'FGPct3_diff_2', 'FTPct_diff_2', 'AstPct_diff_2', 'AstTO_diff_2', 'sos_Score_pg_o_2', 'sos_Score_poss_o_2', 'sos_FGM_pg_o_2', 'sos_FGM_poss_o_2', 'sos_FGA_pg_o_2', 'sos_FGA_poss_o_2', 'sos_FGM3_pg_o_2', 'sos_FGM3_poss_o_2', 'sos_FGA3_pg_o_2', 'sos_FGA3_poss_o_2', 'sos_FTM_pg_o_2', 'sos_FTM_poss_o_2', 'sos_FTA_pg_o_2', 'sos_FTA_poss_o_2', 'sos_OR_pg_o_2', 'sos_OR_poss_o_2', 'sos_DR_pg_o_2', 'sos_DR_poss_o_2', 'sos_Ast_pg_o_2', 'sos_Ast_poss_o_2', 'sos_TO_pg_o_2', 'sos_TO_poss_o_2', 'sos_Stl_pg_o_2', 'sos_Stl_poss_o_2', 'sos_Blk_pg_o_2', 'sos_Blk_poss_o_2', 'sos_PF_pg_o_2', 'sos_PF_poss_o_2', 'sos_Poss_pg_o_2', 'sos_Score_pg_d_2', 'sos_Score_poss_d_2', 'sos_FGM_pg_d_2', 'sos_FGM_poss_d_2', 'sos_FGA_pg_d_2', 'sos_FGA_poss_d_2', 'sos_FGM3_pg_d_2', 'sos_FGM3_poss_d_2', 'sos_FGA3_pg_d_2', 'sos_FGA3_poss_d_2', 'sos_FTM_pg_d_2', 'sos_FTM_poss_d_2', 'sos_FTA_pg_d_2', 'sos_FTA_poss_d_2', 'sos_OR_pg_d_2', 'sos_OR_poss_d_2', 'sos_DR_pg_d_2', 'sos_DR_poss_d_2', 'sos_Ast_pg_d_2', 'sos_Ast_poss_d_2', 'sos_TO_pg_d_2', 'sos_TO_poss_d_2', 'sos_Stl_pg_d_2', 'sos_Stl_poss_d_2', 'sos_Blk_pg_d_2', 'sos_Blk_poss_d_2', 'sos_PF_pg_d_2', 'sos_PF_poss_d_2', 'sos_Poss_pg_d_2', 'sos_FGPct_o_2', 'sos_FGPct3_o_2', 'sos_FTPct_o_2', 'sos_AstPct_o_2', 'sos_AstTO_o_2', 'sos_FGPct_d_2', 'sos_FGPct3_d_2', 'sos_FTPct_d_2', 'sos_AstPct_d_2', 'sos_AstTO_d_2', 'sos_Score_pg_diff_2', 'sos_Score_poss_diff_2', 'sos_FGM_pg_diff_2', 'sos_FGM_poss_diff_2', 'sos_FGA_pg_diff_2', 'sos_FGA_poss_diff_2', 'sos_FGM3_pg_diff_2', 'sos_FGM3_poss_diff_2', 'sos_FGA3_pg_diff_2', 'sos_FGA3_poss_diff_2', 'sos_FTM_pg_diff_2', 'sos_FTM_poss_diff_2', 'sos_FTA_pg_diff_2', 'sos_FTA_poss_diff_2', 'sos_OR_pg_diff_2', 'sos_OR_poss_diff_2', 'sos_DR_pg_diff_2', 'sos_DR_poss_diff_2', 'sos_Ast_pg_diff_2', 'sos_Ast_poss_diff_2', 'sos_TO_pg_diff_2', 'sos_TO_poss_diff_2', 'sos_Stl_pg_diff_2', 'sos_Stl_poss_diff_2', 'sos_Blk_pg_diff_2', 'sos_Blk_poss_diff_2', 'sos_PF_pg_diff_2', 'sos_PF_poss_diff_2', 'sos_Poss_pg_diff_2', 'sos_FGPct_diff_2', 'sos_FGPct3_diff_2', 'sos_FTPct_diff_2', 'sos_AstPct_diff_2', 'sos_AstTO_diff_2']



```python
X_df = train.drop(columns=["Season", "DayNum", "TeamID_1", "TeamID_2", "Margin"])
print(f"X_df: {str(X_df.shape):>24}")

X = torch.as_tensor(
    StandardScaler().fit_transform(X_df.values),
    dtype=torch.float32,
    device=device,
)

print(f"X:    {X.shape}")

y_s = train["Margin"]
print(f"y_s: {str(y_s.shape):>22}")
scaler_y = StandardScaler()

y = torch.tensor(
    scaler_y.fit_transform(train[["Margin"]]).flatten(),
    dtype=torch.float32,
    device=device,
)

print(f"y:    {y.shape}")
```

    X_df:            (202033, 408)
    X:    torch.Size([202033, 408])
    y_s:              (202033,)
    y:    torch.Size([202033])



```python
def brier_score(y_pred_np, y_true_s):
    pred_win_prob = 1 / (1 + np.exp(-y_pred_np * 0.1))
    team_1_won = (y_true_s.values > 0).astype(float)
    return np.mean((pred_win_prob - team_1_won) ** 2)
```


```python
params = {
    "tree_method": "hist",
    "device": "gpu",
    "max_depth": 3,
    "colsample_bytree": 0.5,
    "subsample": 0.8,
    "eta": 0.02,
    "min_child_weight": 80,
    "verbosity": 1,
}

print(f"xgboost")
y_pred_oof = np.zeros(y_s.shape[0])
y_pred_oof2 = np.zeros(y_s.shape[0])
    
for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X_df.index), 1):
    print(f"  fold {fold_n}")
    dm_fold = xgb.DMatrix(X_df.iloc[i_fold], label=y_s.iloc[i_fold])
    dm_oof = xgb.DMatrix(X_df.iloc[i_oof], label=y_s.iloc[i_oof])

    print("  xgb.train")
    m = xgb.train(
        params,
        dm_fold,
        num_boost_round=2000,
        evals=[(dm_fold, "fold"), (dm_oof, "oof")],
        verbose_eval=250,
    )

    y_pred_oof[i_oof] = m.predict(dm_oof)
    
    print("  XGBRegressor")
    m = xgb.XGBRegressor(
        tree_method="hist",
        device="cuda",
        max_depth=3,
        colsample_bytree=0.5,
        subsample=0.8,
        n_estimators=2000,
        learning_rate=0.02,
        min_child_weight=80,
        verbosity=1,
    )
    
    X_fold = cudf.DataFrame.from_pandas(X_df.iloc[i_fold])
    y_fold = cudf.Series(y_s.iloc[i_fold])
    X_oof = cudf.DataFrame.from_pandas(X_df.iloc[i_oof])
    y_oof = cudf.Series(y_s.iloc[i_oof])
    
    m.fit(
        X_fold,
        y_fold,
        verbose=250,
        eval_set=[
            (X_fold, y_fold),
            (X_oof, y_oof)
        ],
    )
    
    y_pred_oof2[i_oof] = m.predict(X_oof)
    
    print()

score = brier_score(y_pred_oof, y_s)
print(f"  score: {score:.4f}")
score = brier_score(y_pred_oof2, y_s)
print(f"  score: {score:.4f}")
```

    xgboost
      fold 1
      xgb.train
    [0]	fold-rmse:16.36785	oof-rmse:16.44650
    [250]	fold-rmse:11.13632	oof-rmse:11.25991
    [500]	fold-rmse:10.89192	oof-rmse:11.03729
    [750]	fold-rmse:10.84072	oof-rmse:11.00809
    [1000]	fold-rmse:10.81164	oof-rmse:11.00183
    [1250]	fold-rmse:10.78834	oof-rmse:11.00055
    [1500]	fold-rmse:10.76690	oof-rmse:11.00130
    [1750]	fold-rmse:10.74744	oof-rmse:11.00363
    [1999]	fold-rmse:10.72917	oof-rmse:11.00673
      XGBRegressor
    [0]	validation_0-rmse:16.36785	validation_1-rmse:16.44650
    [250]	validation_0-rmse:11.13632	validation_1-rmse:11.25991
    [500]	validation_0-rmse:10.89192	validation_1-rmse:11.03729
    [750]	validation_0-rmse:10.84072	validation_1-rmse:11.00809
    [1000]	validation_0-rmse:10.81164	validation_1-rmse:11.00183
    [1250]	validation_0-rmse:10.78834	validation_1-rmse:11.00055
    [1500]	validation_0-rmse:10.76690	validation_1-rmse:11.00130
    [1750]	validation_0-rmse:10.74744	validation_1-rmse:11.00363
    [1999]	validation_0-rmse:10.72917	validation_1-rmse:11.00673
    
      fold 2
      xgb.train
    [0]	fold-rmse:16.37827	oof-rmse:16.40540
    [250]	fold-rmse:11.15190	oof-rmse:11.15731
    [500]	fold-rmse:10.90885	oof-rmse:10.93807
    [750]	fold-rmse:10.85732	oof-rmse:10.91094
    [1000]	fold-rmse:10.82878	oof-rmse:10.90627
    [1250]	fold-rmse:10.80507	oof-rmse:10.90547
    [1500]	fold-rmse:10.78433	oof-rmse:10.90767
    [1750]	fold-rmse:10.76528	oof-rmse:10.91017
    [1999]	fold-rmse:10.74664	oof-rmse:10.91299
      XGBRegressor
    [0]	validation_0-rmse:16.37827	validation_1-rmse:16.40540
    [250]	validation_0-rmse:11.15190	validation_1-rmse:11.15731
    [500]	validation_0-rmse:10.90885	validation_1-rmse:10.93807
    [750]	validation_0-rmse:10.85732	validation_1-rmse:10.91094
    [1000]	validation_0-rmse:10.82878	validation_1-rmse:10.90627
    [1250]	validation_0-rmse:10.80507	validation_1-rmse:10.90547
    [1500]	validation_0-rmse:10.78433	validation_1-rmse:10.90767
    [1750]	validation_0-rmse:10.76528	validation_1-rmse:10.91017
    [1999]	validation_0-rmse:10.74664	validation_1-rmse:10.91299
    
      fold 3
      xgb.train
    [0]	fold-rmse:16.38887	oof-rmse:16.36117
    [250]	fold-rmse:11.13991	oof-rmse:11.21294
    [500]	fold-rmse:10.89497	oof-rmse:10.99988
    [750]	fold-rmse:10.84395	oof-rmse:10.97439
    [1000]	fold-rmse:10.81500	oof-rmse:10.97160
    [1250]	fold-rmse:10.79166	oof-rmse:10.97178
    [1500]	fold-rmse:10.77153	oof-rmse:10.97361
    [1750]	fold-rmse:10.75255	oof-rmse:10.97509
    [1999]	fold-rmse:10.73453	oof-rmse:10.97778
      XGBRegressor
    [0]	validation_0-rmse:16.38887	validation_1-rmse:16.36117
    [250]	validation_0-rmse:11.13991	validation_1-rmse:11.21294
    [500]	validation_0-rmse:10.89497	validation_1-rmse:10.99988
    [750]	validation_0-rmse:10.84395	validation_1-rmse:10.97439
    [1000]	validation_0-rmse:10.81500	validation_1-rmse:10.97160
    [1250]	validation_0-rmse:10.79166	validation_1-rmse:10.97178
    [1500]	validation_0-rmse:10.77153	validation_1-rmse:10.97361
    [1750]	validation_0-rmse:10.75255	validation_1-rmse:10.97509
    [1999]	validation_0-rmse:10.73453	validation_1-rmse:10.97778
    
      fold 4
      xgb.train
    [0]	fold-rmse:16.40045	oof-rmse:16.31378
    [250]	fold-rmse:11.14824	oof-rmse:11.17067
    [500]	fold-rmse:10.90203	oof-rmse:10.96252
    [750]	fold-rmse:10.85212	oof-rmse:10.93718
    [1000]	fold-rmse:10.82359	oof-rmse:10.93289
    [1250]	fold-rmse:10.80032	oof-rmse:10.93207
    [1500]	fold-rmse:10.77977	oof-rmse:10.93305
    [1750]	fold-rmse:10.76021	oof-rmse:10.93483
    [1999]	fold-rmse:10.74186	oof-rmse:10.93748
      XGBRegressor
    [0]	validation_0-rmse:16.40045	validation_1-rmse:16.31378
    [250]	validation_0-rmse:11.14824	validation_1-rmse:11.17067
    [500]	validation_0-rmse:10.90203	validation_1-rmse:10.96252
    [750]	validation_0-rmse:10.85212	validation_1-rmse:10.93718
    [1000]	validation_0-rmse:10.82359	validation_1-rmse:10.93289
    [1250]	validation_0-rmse:10.80032	validation_1-rmse:10.93207
    [1500]	validation_0-rmse:10.77977	validation_1-rmse:10.93305
    [1750]	validation_0-rmse:10.76021	validation_1-rmse:10.93483
    [1999]	validation_0-rmse:10.74186	validation_1-rmse:10.93748
    
      fold 5
      xgb.train
    [0]	fold-rmse:16.38079	oof-rmse:16.39268
    [250]	fold-rmse:11.14486	oof-rmse:11.19691
    [500]	fold-rmse:10.89887	oof-rmse:10.97839
    [750]	fold-rmse:10.84785	oof-rmse:10.95338
    [1000]	fold-rmse:10.81966	oof-rmse:10.94925
    [1250]	fold-rmse:10.79644	oof-rmse:10.94937
    [1500]	fold-rmse:10.77548	oof-rmse:10.95056
    [1750]	fold-rmse:10.75667	oof-rmse:10.95349
    [1999]	fold-rmse:10.73897	oof-rmse:10.95602
      XGBRegressor
    [0]	validation_0-rmse:16.38079	validation_1-rmse:16.39268
    [250]	validation_0-rmse:11.14486	validation_1-rmse:11.19691
    [500]	validation_0-rmse:10.89887	validation_1-rmse:10.97839
    [750]	validation_0-rmse:10.84785	validation_1-rmse:10.95338
    [1000]	validation_0-rmse:10.81966	validation_1-rmse:10.94925
    [1250]	validation_0-rmse:10.79644	validation_1-rmse:10.94937
    [1500]	validation_0-rmse:10.77548	validation_1-rmse:10.95056
    [1750]	validation_0-rmse:10.75667	validation_1-rmse:10.95349
    [1999]	validation_0-rmse:10.73897	validation_1-rmse:10.95602
    
      score: 0.1657
      score: 0.1657



```python
print("torch")
n_epochs = 1_000
hidden_size = 64
loss_fn = torch.nn.MSELoss()

y_pred_oof = torch.zeros(
    y.shape[0],
    dtype=torch.float32,
    requires_grad=False,
    device=device,
)

for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X_df.index), 1):
    print(f"  fold {fold_n}")

    weights1 = torch.nn.Parameter(
        0.1 * torch.randn(X_df.shape[1], hidden_size, device=device)
    )
    bias1 = torch.nn.Parameter(torch.zeros(hidden_size, device=device))
    weights2 = torch.nn.Parameter(0.1 * torch.randn(hidden_size, 1, device=device))
    bias2 = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.Adam([weights1, bias1, weights2, bias2], weight_decay=1e-4)

    for epoch_n in range(1, n_epochs + 1):
        y_pred_fold_epoch = F.leaky_relu(X[i_fold] @ weights1 + bias1, negative_slope=0.1) @ weights2 + bias2
        loss_fold_epoch = loss_fn(y_pred_fold_epoch, y[i_fold].view(-1, 1))
        optimizer.zero_grad()
        loss_fold_epoch.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_oof_epoch = F.leaky_relu(X[i_oof] @ weights1 + bias1, negative_slope=0.1) @ weights2 + bias2
            loss_oof_epoch = loss_fn(y_pred_oof_epoch, y[i_oof].view(-1, 1))

        if epoch_n > (n_epochs - 3):
            print(
                f"    epoch {epoch_n:>6}: "
                f"fold={loss_fold_epoch.item():.4f} "
                f"oof={loss_oof_epoch.item():.4f}"
            )

    with torch.no_grad():
        y_pred_oof[i_oof] = (
            F.leaky_relu(X[i_oof] @ weights1 + bias1, negative_slope=0.1) @ weights2 + bias2
        ).flatten()

    print()

y_pred_oof = scaler_y.inverse_transform(
    y_pred_oof.cpu().numpy().reshape(-1, 1)
).flatten()

score = brier_score(y_pred_oof, y_s)
print(f"  score: {score:.4f}")
```

    torch
      fold 1
        epoch    998: fold=0.4277 oof=0.4486
        epoch    999: fold=0.4277 oof=0.4486
        epoch   1000: fold=0.4276 oof=0.4486
    
      fold 2
        epoch    998: fold=0.4271 oof=0.4424
        epoch    999: fold=0.4271 oof=0.4424
        epoch   1000: fold=0.4271 oof=0.4424
    
      fold 3
        epoch    998: fold=0.4231 oof=0.4471
        epoch    999: fold=0.4231 oof=0.4474
        epoch   1000: fold=0.4231 oof=0.4470
    
      fold 4
        epoch    998: fold=0.4265 oof=0.4440
        epoch    999: fold=0.4265 oof=0.4439
        epoch   1000: fold=0.4265 oof=0.4439
    
      fold 5
        epoch    998: fold=0.4277 oof=0.4434
        epoch    999: fold=0.4276 oof=0.4434
        epoch   1000: fold=0.4276 oof=0.4434
    
      score: 0.1655



```python

```
