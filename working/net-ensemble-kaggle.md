```python
import os

IS_KAGGLE = bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE'))
print(f"running on kaggle: {IS_KAGGLE}")
```

    running on kaggle: False



```python
if not IS_KAGGLE:
    !pip install --upgrade numpy pandas xgboost scikit-learn
    !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    !pip install \
        --extra-index-url=https://pypi.nvidia.com \
        "cudf-cu12==25.2.*" "dask-cudf-cu12==25.2.*" "cuml-cu12==25.2.*" \
        "cugraph-cu12==25.2.*" "nx-cugraph-cu12==25.2.*" "cuspatial-cu12==25.2.*" \
        "cuproj-cu12==25.2.*" "cuxfilter-cu12==25.2.*" "cucim-cu12==25.2.*" \
        "pylibraft-cu12==25.2.*" "raft-dask-cu12==25.2.*" "cuvs-cu12==25.2.*" \
        "nx-cugraph-cu12==25.2.*"

!pip install gputil
```

    Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (1.26.3)
    Collecting numpy
      Downloading numpy-2.2.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m62.0/62.0 kB[0m [31m2.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (2.2.0)
    Collecting pandas
      Downloading pandas-2.2.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (89 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m89.9/89.9 kB[0m [31m11.5 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: xgboost in /usr/local/lib/python3.11/dist-packages (1.7.6)
    Collecting xgboost
      Downloading xgboost-2.1.4-py3-none-manylinux_2_28_x86_64.whl.metadata (2.1 kB)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.11/dist-packages (1.3.0)
    Collecting scikit-learn
      Downloading scikit_learn-1.6.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (18 kB)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/lib/python3/dist-packages (from pandas) (2022.1)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas) (2023.4)
    Collecting nvidia-nccl-cu12 (from xgboost)
      Downloading nvidia_nccl_cu12-2.25.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
    Requirement already satisfied: scipy in /usr/local/lib/python3.11/dist-packages (from xgboost) (1.11.2)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn) (1.3.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn) (3.2.0)
    Requirement already satisfied: six>=1.5 in /usr/lib/python3/dist-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
    Collecting numpy
      Downloading numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m61.0/61.0 kB[0m [31m13.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pandas-2.2.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.1/13.1 MB[0m [31m93.4 MB/s[0m eta [36m0:00:00[0m:00:01[0m:01[0m
    [?25hDownloading xgboost-2.1.4-py3-none-manylinux_2_28_x86_64.whl (223.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m223.6/223.6 MB[0m [31m15.3 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hDownloading scikit_learn-1.6.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.5/13.5 MB[0m [31m110.4 MB/s[0m eta [36m0:00:00[0m00:01[0m0:01[0m
    [?25hDownloading numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m18.3/18.3 MB[0m [31m113.8 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hDownloading nvidia_nccl_cu12-2.25.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (201.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m201.4/201.4 MB[0m [31m18.1 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hInstalling collected packages: nvidia-nccl-cu12, numpy, pandas, xgboost, scikit-learn
      Attempting uninstall: numpy
        Found existing installation: numpy 1.26.3
        Uninstalling numpy-1.26.3:
          Successfully uninstalled numpy-1.26.3
      Attempting uninstall: pandas
        Found existing installation: pandas 2.2.0
        Uninstalling pandas-2.2.0:
          Successfully uninstalled pandas-2.2.0
      Attempting uninstall: xgboost
        Found existing installation: xgboost 1.7.6
        Uninstalling xgboost-1.7.6:
          Successfully uninstalled xgboost-1.7.6
      Attempting uninstall: scikit-learn
        Found existing installation: scikit-learn 1.3.0
        Uninstalling scikit-learn-1.3.0:
          Successfully uninstalled scikit-learn-1.3.0
    Successfully installed numpy-1.26.4 nvidia-nccl-cu12-2.25.1 pandas-2.2.3 scikit-learn-1.6.1 xgboost-2.1.4
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
    Collecting cudf-cu12==25.2.*
      Downloading https://pypi.nvidia.com/cudf-cu12/cudf_cu12-25.2.2-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (2.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.2/2.2 MB[0m [31m74.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting dask-cudf-cu12==25.2.*
      Downloading https://pypi.nvidia.com/dask-cudf-cu12/dask_cudf_cu12-25.2.2-py3-none-any.whl (50 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m50.4/50.4 kB[0m [31m98.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting cuml-cu12==25.2.*
      Downloading https://pypi.nvidia.com/cuml-cu12/cuml_cu12-25.2.1-cp311-cp311-manylinux_2_28_x86_64.whl (9.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m9.7/9.7 MB[0m [31m107.1 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting cugraph-cu12==25.2.*
      Downloading https://pypi.nvidia.com/cugraph-cu12/cugraph_cu12-25.2.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (3.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.2/3.2 MB[0m [31m264.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nx-cugraph-cu12==25.2.*
      Downloading https://pypi.nvidia.com/nx-cugraph-cu12/nx_cugraph_cu12-25.2.0-py3-none-any.whl (160 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m160.2/160.2 kB[0m [31m157.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting cuspatial-cu12==25.2.*
      Downloading https://pypi.nvidia.com/cuspatial-cu12/cuspatial_cu12-25.2.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (4.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.3/4.3 MB[0m [31m58.7 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting cuproj-cu12==25.2.*
      Downloading https://pypi.nvidia.com/cuproj-cu12/cuproj_cu12-25.2.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (1.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m98.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting cuxfilter-cu12==25.2.*
      Downloading https://pypi.nvidia.com/cuxfilter-cu12/cuxfilter_cu12-25.2.0-py3-none-any.whl (83 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m83.6/83.6 kB[0m [31m121.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting cucim-cu12==25.2.*
      Downloading https://pypi.nvidia.com/cucim-cu12/cucim_cu12-25.2.0-cp311-cp311-manylinux_2_28_x86_64.whl (5.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.6/5.6 MB[0m [31m36.6 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting pylibraft-cu12==25.2.*
      Downloading https://pypi.nvidia.com/pylibraft-cu12/pylibraft_cu12-25.2.0-cp311-cp311-manylinux_2_28_x86_64.whl (851 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m851.2/851.2 kB[0m [31m251.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting raft-dask-cu12==25.2.*
      Downloading https://pypi.nvidia.com/raft-dask-cu12/raft_dask_cu12-25.2.0-cp311-cp311-manylinux_2_28_x86_64.whl (293.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m293.5/293.5 MB[0m [31m102.0 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting cuvs-cu12==25.2.*
      Downloading https://pypi.nvidia.com/cuvs-cu12/cuvs_cu12-25.2.1-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (2.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.3/2.3 MB[0m [31m160.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: cachetools in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (5.3.2)
    Collecting cuda-python<13.0a0,>=12.6.2 (from cudf-cu12==25.2.*)
      Downloading cuda_python-12.8.0-py3-none-any.whl.metadata (15 kB)
    Requirement already satisfied: cupy-cuda12x>=12.0.0 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (12.2.0)
    Requirement already satisfied: fsspec>=0.6.0 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (2023.6.0)
    Collecting libcudf-cu12==25.2.* (from cudf-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/libcudf-cu12/libcudf_cu12-25.2.2-py3-none-manylinux_2_28_x86_64.whl (557.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m557.7/557.7 MB[0m [31m26.1 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting numba-cuda<0.3.0a0,>=0.2.0 (from cudf-cu12==25.2.*)
      Downloading numba_cuda-0.2.0-py3-none-any.whl.metadata (1.5 kB)
    Collecting numba<0.61.0a0,>=0.59.1 (from cudf-cu12==25.2.*)
      Downloading numba-0.60.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.7 kB)
    Requirement already satisfied: numpy<3.0a0,>=1.23 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (1.26.4)
    Collecting nvtx>=0.2.1 (from cudf-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/nvtx/nvtx-0.2.11-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (527 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m527.9/527.9 kB[0m [31m273.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (23.2)
    Requirement already satisfied: pandas<2.2.4dev0,>=2.0 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (2.2.3)
    Requirement already satisfied: pyarrow<20.0.0a0,>=14.0.0 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (15.0.0)
    Collecting pylibcudf-cu12==25.2.* (from cudf-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/pylibcudf-cu12/pylibcudf_cu12-25.2.2-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (27.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m27.3/27.3 MB[0m [31m31.6 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting pynvjitlink-cu12 (from cudf-cu12==25.2.*)
      Downloading pynvjitlink_cu12-0.5.2-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (1.5 kB)
    Collecting rich (from cudf-cu12==25.2.*)
      Downloading rich-13.9.4-py3-none-any.whl.metadata (18 kB)
    Collecting rmm-cu12==25.2.* (from cudf-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/rmm-cu12/rmm_cu12-25.2.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (2.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.4/2.4 MB[0m [31m53.8 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: typing_extensions>=4.0.0 in /usr/local/lib/python3.11/dist-packages (from cudf-cu12==25.2.*) (4.9.0)
    Collecting pynvml<13.0.0a0,>=12.0.0 (from dask-cudf-cu12==25.2.*)
      Downloading pynvml-12.0.0-py3-none-any.whl.metadata (5.4 kB)
    Collecting rapids-dask-dependency==25.2.* (from dask-cudf-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/rapids-dask-dependency/rapids_dask_dependency-25.2.0-py3-none-any.whl (22 kB)
    Collecting dask-cuda==25.2.* (from cuml-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/dask-cuda/dask_cuda-25.2.0-py3-none-any.whl (133 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m133.9/133.9 kB[0m [31m178.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.11/dist-packages (from cuml-cu12==25.2.*) (1.3.2)
    Collecting libcuml-cu12==25.2.* (from cuml-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/libcuml-cu12/libcuml_cu12-25.2.1-py3-none-manylinux_2_28_x86_64.whl (405.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m405.0/405.0 MB[0m [31m51.3 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cublas-cu12 (from cuml-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/nvidia-cublas-cu12/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl (594.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m594.3/594.3 MB[0m [31m51.1 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cufft-cu12 (from cuml-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/nvidia-cufft-cu12/nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (193.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m193.1/193.1 MB[0m [31m92.5 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-curand-cu12 (from cuml-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/nvidia-curand-cu12/nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl (63.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m63.6/63.6 MB[0m [31m82.8 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cusolver-cu12 (from cuml-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/nvidia-cusolver-cu12/nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl (267.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m267.5/267.5 MB[0m [31m72.6 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting nvidia-cusparse-cu12 (from cuml-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/nvidia-cusparse-cu12/nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (288.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m288.2/288.2 MB[0m [31m80.3 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: scipy>=1.8.0 in /usr/local/lib/python3.11/dist-packages (from cuml-cu12==25.2.*) (1.11.2)
    Collecting treelite==4.4.1 (from cuml-cu12==25.2.*)
      Downloading treelite-4.4.1-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting libcugraph-cu12==25.2.* (from cugraph-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/libcugraph-cu12/libcugraph_cu12-25.2.0-py3-none-manylinux_2_28_x86_64.whl (1425.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.4/1.4 GB[0m [31m24.0 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hCollecting pylibcugraph-cu12==25.2.* (from cugraph-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/pylibcugraph-cu12/pylibcugraph_cu12-25.2.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (2.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m43.5 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting ucx-py-cu12==0.42.* (from cugraph-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/ucx-py-cu12/ucx_py_cu12-0.42.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl (2.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.3/2.3 MB[0m [31m157.5 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: networkx>=3.2 in /usr/local/lib/python3.11/dist-packages (from nx-cugraph-cu12==25.2.*) (3.2.1)
    Collecting geopandas>=1.0.0 (from cuspatial-cu12==25.2.*)
      Downloading geopandas-1.0.1-py3-none-any.whl.metadata (2.2 kB)
    Collecting libcuspatial-cu12==25.2.* (from cuspatial-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/libcuspatial-cu12/libcuspatial_cu12-25.2.0-py3-none-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (32.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m32.6/32.6 MB[0m [31m145.9 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting bokeh>=3.1 (from cuxfilter-cu12==25.2.*)
      Downloading bokeh-3.6.3-py3-none-any.whl.metadata (12 kB)
    Collecting datashader>=0.15 (from cuxfilter-cu12==25.2.*)
      Downloading datashader-0.17.0-py3-none-any.whl.metadata (7.6 kB)
    Collecting holoviews>=1.16.0 (from cuxfilter-cu12==25.2.*)
      Downloading holoviews-1.20.1-py3-none-any.whl.metadata (9.9 kB)
    Collecting jupyter-server-proxy (from cuxfilter-cu12==25.2.*)
      Downloading jupyter_server_proxy-4.4.0-py3-none-any.whl.metadata (8.7 kB)
    Collecting panel>=1.0 (from cuxfilter-cu12==25.2.*)
      Downloading panel-1.6.1-py3-none-any.whl.metadata (15 kB)
    Requirement already satisfied: click in /usr/local/lib/python3.11/dist-packages (from cucim-cu12==25.2.*) (8.1.7)
    Collecting lazy-loader>=0.4 (from cucim-cu12==25.2.*)
      Downloading lazy_loader-0.4-py3-none-any.whl.metadata (7.6 kB)
    Requirement already satisfied: scikit-image<0.26.0a0,>=0.19.0 in /usr/local/lib/python3.11/dist-packages (from cucim-cu12==25.2.*) (0.21.0)
    Collecting libraft-cu12==25.2.* (from pylibraft-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/libraft-cu12/libraft_cu12-25.2.0-py3-none-manylinux_2_28_x86_64.whl (22.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m22.3/22.3 MB[0m [31m65.6 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting distributed-ucxx-cu12==0.42.* (from raft-dask-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/distributed-ucxx-cu12/distributed_ucxx_cu12-0.42.0-py3-none-any.whl (24 kB)
    Collecting libcuvs-cu12==25.2.* (from cuvs-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/libcuvs-cu12/libcuvs_cu12-25.2.1-py3-none-manylinux_2_28_x86_64.whl (1184.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 GB[0m [31m18.8 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting zict>=2.0.0 (from dask-cuda==25.2.*->cuml-cu12==25.2.*)
      Downloading zict-3.0.0-py2.py3-none-any.whl.metadata (899 bytes)
    Collecting ucxx-cu12==0.42.* (from distributed-ucxx-cu12==0.42.*->raft-dask-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/ucxx-cu12/ucxx_cu12-0.42.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (725 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m725.3/725.3 kB[0m [31m269.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting libkvikio-cu12==25.2.* (from libcudf-cu12==25.2.*->cudf-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/libkvikio-cu12/libkvikio_cu12-25.2.1-py3-none-manylinux_2_28_x86_64.whl (2.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.1/2.1 MB[0m [31m117.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting nvidia-nvcomp-cu12==4.2.0.11 (from libcudf-cu12==25.2.*->cudf-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/nvidia-nvcomp-cu12/nvidia_nvcomp_cu12-4.2.0.11-py3-none-manylinux_2_28_x86_64.whl (46.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m46.3/46.3 MB[0m [31m24.7 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting dask==2024.12.1 (from rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*)
      Downloading dask-2024.12.1-py3-none-any.whl.metadata (3.7 kB)
    Collecting distributed==2024.12.1 (from rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*)
      Downloading distributed-2024.12.1-py3-none-any.whl.metadata (3.3 kB)
    Collecting dask-expr==1.1.21 (from rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*)
      Downloading dask_expr-1.1.21-py3-none-any.whl.metadata (2.6 kB)
    Collecting libucx-cu12<1.19,>=1.15.0 (from ucx-py-cu12==0.42.*->cugraph-cu12==25.2.*)
      Downloading libucx_cu12-1.18.0-py3-none-manylinux_2_28_x86_64.whl.metadata (2.9 kB)
    Collecting cloudpickle>=3.0.0 (from dask==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*)
      Downloading cloudpickle-3.1.1-py3-none-any.whl.metadata (7.1 kB)
    Collecting partd>=1.4.0 (from dask==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*)
      Downloading partd-1.4.2-py3-none-any.whl.metadata (4.6 kB)
    Requirement already satisfied: pyyaml>=5.3.1 in /usr/lib/python3/dist-packages (from dask==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (5.4.1)
    Collecting toolz>=0.10.0 (from dask==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*)
      Downloading toolz-1.0.0-py3-none-any.whl.metadata (5.1 kB)
    Collecting importlib_metadata>=4.13.0 (from dask==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*)
      Downloading importlib_metadata-8.6.1-py3-none-any.whl.metadata (4.7 kB)
    Requirement already satisfied: jinja2>=2.10.3 in /usr/local/lib/python3.11/dist-packages (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (3.1.3)
    Collecting locket>=1.0.0 (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*)
      Downloading locket-1.0.0-py2.py3-none-any.whl.metadata (2.8 kB)
    Collecting msgpack>=1.0.2 (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*)
      Downloading msgpack-1.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (8.4 kB)
    Requirement already satisfied: psutil>=5.8.0 in /usr/local/lib/python3.11/dist-packages (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (5.9.8)
    Collecting sortedcontainers>=2.0.5 (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*)
      Downloading sortedcontainers-2.4.0-py2.py3-none-any.whl.metadata (10 kB)
    Collecting tblib>=1.6.0 (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*)
      Downloading tblib-3.0.0-py3-none-any.whl.metadata (25 kB)
    Requirement already satisfied: tornado>=6.2.0 in /usr/local/lib/python3.11/dist-packages (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (6.4)
    Requirement already satisfied: urllib3>=1.26.5 in /usr/local/lib/python3.11/dist-packages (from distributed==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*) (2.0.7)
    Collecting libucxx-cu12==0.42.* (from ucxx-cu12==0.42.*->distributed-ucxx-cu12==0.42.*->raft-dask-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/libucxx-cu12/libucxx_cu12-0.42.0-py3-none-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (514 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m514.8/514.8 kB[0m [31m171.8 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: contourpy>=1.2 in /usr/local/lib/python3.11/dist-packages (from bokeh>=3.1->cuxfilter-cu12==25.2.*) (1.2.0)
    Requirement already satisfied: pillow>=7.1.0 in /usr/local/lib/python3.11/dist-packages (from bokeh>=3.1->cuxfilter-cu12==25.2.*) (9.5.0)
    Collecting xyzservices>=2021.09.1 (from bokeh>=3.1->cuxfilter-cu12==25.2.*)
      Downloading xyzservices-2025.1.0-py3-none-any.whl.metadata (4.3 kB)
    Collecting cuda-bindings~=12.8.0 (from cuda-python<13.0a0,>=12.6.2->cudf-cu12==25.2.*)
      Downloading cuda_bindings-12.8.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (13 kB)
    Requirement already satisfied: fastrlock>=0.5 in /usr/local/lib/python3.11/dist-packages (from cupy-cuda12x>=12.0.0->cudf-cu12==25.2.*) (0.8.2)
    Collecting colorcet (from datashader>=0.15->cuxfilter-cu12==25.2.*)
      Downloading colorcet-3.1.0-py3-none-any.whl.metadata (6.3 kB)
    Collecting multipledispatch (from datashader>=0.15->cuxfilter-cu12==25.2.*)
      Downloading multipledispatch-1.0.0-py3-none-any.whl.metadata (3.8 kB)
    Collecting param (from datashader>=0.15->cuxfilter-cu12==25.2.*)
      Downloading param-2.2.0-py3-none-any.whl.metadata (6.6 kB)
    Collecting pyct (from datashader>=0.15->cuxfilter-cu12==25.2.*)
      Downloading pyct-0.5.0-py2.py3-none-any.whl.metadata (7.4 kB)
    Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from datashader>=0.15->cuxfilter-cu12==25.2.*) (2.31.0)
    Collecting xarray (from datashader>=0.15->cuxfilter-cu12==25.2.*)
      Downloading xarray-2025.1.2-py3-none-any.whl.metadata (11 kB)
    Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /usr/local/lib/python3.11/dist-packages (from fsspec[http]>=0.6.0->cugraph-cu12==25.2.*) (3.9.1)
    Collecting pyogrio>=0.7.2 (from geopandas>=1.0.0->cuspatial-cu12==25.2.*)
      Downloading pyogrio-0.10.0-cp311-cp311-manylinux_2_28_x86_64.whl.metadata (5.5 kB)
    Collecting pyproj>=3.3.0 (from geopandas>=1.0.0->cuspatial-cu12==25.2.*)
      Downloading pyproj-3.7.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (31 kB)
    Collecting shapely>=2.0.0 (from geopandas>=1.0.0->cuspatial-cu12==25.2.*)
      Downloading shapely-2.0.7-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)
    Collecting pyviz-comms>=2.1 (from holoviews>=1.16.0->cuxfilter-cu12==25.2.*)
      Downloading pyviz_comms-3.0.4-py3-none-any.whl.metadata (7.7 kB)
    Collecting llvmlite<0.44,>=0.43.0dev0 (from numba<0.61.0a0,>=0.59.1->cudf-cu12==25.2.*)
      Downloading llvmlite-0.43.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.8 kB)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas<2.2.4dev0,>=2.0->cudf-cu12==25.2.*) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/lib/python3/dist-packages (from pandas<2.2.4dev0,>=2.0->cudf-cu12==25.2.*) (2022.1)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas<2.2.4dev0,>=2.0->cudf-cu12==25.2.*) (2023.4)
    Requirement already satisfied: bleach in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->cuxfilter-cu12==25.2.*) (6.1.0)
    Collecting linkify-it-py (from panel>=1.0->cuxfilter-cu12==25.2.*)
      Downloading linkify_it_py-2.0.3-py3-none-any.whl.metadata (8.5 kB)
    Requirement already satisfied: markdown in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->cuxfilter-cu12==25.2.*) (3.5.2)
    Collecting markdown-it-py (from panel>=1.0->cuxfilter-cu12==25.2.*)
      Downloading markdown_it_py-3.0.0-py3-none-any.whl.metadata (6.9 kB)
    Collecting mdit-py-plugins (from panel>=1.0->cuxfilter-cu12==25.2.*)
      Downloading mdit_py_plugins-0.4.2-py3-none-any.whl.metadata (2.8 kB)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from panel>=1.0->cuxfilter-cu12==25.2.*) (4.66.1)
    Collecting nvidia-ml-py<13.0.0a0,>=12.0.0 (from pynvml<13.0.0a0,>=12.0.0->dask-cudf-cu12==25.2.*)
      Downloading nvidia_ml_py-12.570.86-py3-none-any.whl.metadata (8.7 kB)
    Requirement already satisfied: imageio>=2.27 in /usr/local/lib/python3.11/dist-packages (from scikit-image<0.26.0a0,>=0.19.0->cucim-cu12==25.2.*) (2.33.1)
    Requirement already satisfied: tifffile>=2022.8.12 in /usr/local/lib/python3.11/dist-packages (from scikit-image<0.26.0a0,>=0.19.0->cucim-cu12==25.2.*) (2023.12.9)
    Requirement already satisfied: PyWavelets>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from scikit-image<0.26.0a0,>=0.19.0->cucim-cu12==25.2.*) (1.5.0)
    Requirement already satisfied: jupyter-server>=1.24.0 in /usr/local/lib/python3.11/dist-packages (from jupyter-server-proxy->cuxfilter-cu12==25.2.*) (2.12.5)
    Collecting simpervisor>=1.0.0 (from jupyter-server-proxy->cuxfilter-cu12==25.2.*)
      Downloading simpervisor-1.0.0-py3-none-any.whl.metadata (4.3 kB)
    Requirement already satisfied: traitlets>=5.1.0 in /usr/local/lib/python3.11/dist-packages (from jupyter-server-proxy->cuxfilter-cu12==25.2.*) (5.14.1)
    Collecting nvidia-nvjitlink-cu12 (from nvidia-cufft-cu12->cuml-cu12==25.2.*)
      Downloading https://pypi.nvidia.com/nvidia-nvjitlink-cu12/nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (39.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m39.3/39.3 MB[0m [31m111.4 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.11/dist-packages (from rich->cudf-cu12==25.2.*) (2.17.2)
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
    Collecting mdurl~=0.1 (from markdown-it-py->panel>=1.0->cuxfilter-cu12==25.2.*)
      Downloading mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
    Requirement already satisfied: certifi in /usr/lib/python3/dist-packages (from pyogrio>=0.7.2->geopandas>=1.0.0->cuspatial-cu12==25.2.*) (2020.6.20)
    Requirement already satisfied: six>=1.5 in /usr/lib/python3/dist-packages (from python-dateutil>=2.8.2->pandas<2.2.4dev0,>=2.0->cudf-cu12==25.2.*) (1.16.0)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.11/dist-packages (from bleach->panel>=1.0->cuxfilter-cu12==25.2.*) (0.5.1)
    Collecting uc-micro-py (from linkify-it-py->panel>=1.0->cuxfilter-cu12==25.2.*)
      Downloading uc_micro_py-1.0.3-py3-none-any.whl.metadata (2.0 kB)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->datashader>=0.15->cuxfilter-cu12==25.2.*) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/lib/python3/dist-packages (from requests->datashader>=0.15->cuxfilter-cu12==25.2.*) (3.3)
    Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.11/dist-packages (from anyio>=3.1.0->jupyter-server>=1.24.0->jupyter-server-proxy->cuxfilter-cu12==25.2.*) (1.3.0)
    Collecting zipp>=3.20 (from importlib_metadata>=4.13.0->dask==2024.12.1->rapids-dask-dependency==25.2.*->dask-cudf-cu12==25.2.*)
      Downloading zipp-3.21.0-py3-none-any.whl.metadata (3.7 kB)
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
    Downloading treelite-4.4.1-py3-none-manylinux2014_x86_64.whl (922 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m922.8/922.8 kB[0m [31m67.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading dask-2024.12.1-py3-none-any.whl (1.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m116.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading dask_expr-1.1.21-py3-none-any.whl (244 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m244.3/244.3 kB[0m [31m54.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading distributed-2024.12.1-py3-none-any.whl (1.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m113.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading bokeh-3.6.3-py3-none-any.whl (6.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.9/6.9 MB[0m [31m98.4 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading cuda_python-12.8.0-py3-none-any.whl (11 kB)
    Downloading datashader-0.17.0-py3-none-any.whl (18.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m18.3/18.3 MB[0m [31m89.3 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading geopandas-1.0.1-py3-none-any.whl (323 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m323.6/323.6 kB[0m [31m46.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading holoviews-1.20.1-py3-none-any.whl (5.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.0/5.0 MB[0m [31m135.9 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hDownloading lazy_loader-0.4-py3-none-any.whl (12 kB)
    Downloading numba-0.60.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.7/3.7 MB[0m [31m114.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading numba_cuda-0.2.0-py3-none-any.whl (443 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m443.7/443.7 kB[0m [31m70.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading panel-1.6.1-py3-none-any.whl (28.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m28.0/28.0 MB[0m [31m68.1 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading pynvml-12.0.0-py3-none-any.whl (26 kB)
    Downloading jupyter_server_proxy-4.4.0-py3-none-any.whl (37 kB)
    Downloading pynvjitlink_cu12-0.5.2-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (46.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m46.2/46.2 MB[0m [31m62.7 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading rich-13.9.4-py3-none-any.whl (242 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m242.4/242.4 kB[0m [31m50.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading cuda_bindings-12.8.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (11.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.3/11.3 MB[0m [31m117.0 MB/s[0m eta [36m0:00:00[0m00:01[0m0:01[0m
    [?25hDownloading libucx_cu12-1.18.0-py3-none-manylinux_2_28_x86_64.whl (27.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m27.5/27.5 MB[0m [31m74.0 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading llvmlite-0.43.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (43.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m43.9/43.9 MB[0m [31m61.1 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading markdown_it_py-3.0.0-py3-none-any.whl (87 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m87.5/87.5 kB[0m [31m22.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_ml_py-12.570.86-py3-none-any.whl (44 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m44.4/44.4 kB[0m [31m10.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading param-2.2.0-py3-none-any.whl (119 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m119.0/119.0 kB[0m [31m27.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pyogrio-0.10.0-cp311-cp311-manylinux_2_28_x86_64.whl (24.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m24.1/24.1 MB[0m [31m94.1 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading pyproj-3.7.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (9.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m9.5/9.5 MB[0m [31m163.6 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hDownloading pyviz_comms-3.0.4-py3-none-any.whl (83 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m83.8/83.8 kB[0m [31m21.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading shapely-2.0.7-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.5/2.5 MB[0m [31m109.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading simpervisor-1.0.0-py3-none-any.whl (8.3 kB)
    Downloading toolz-1.0.0-py3-none-any.whl (56 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.4/56.4 kB[0m [31m14.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading xyzservices-2025.1.0-py3-none-any.whl (88 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m88.4/88.4 kB[0m [31m22.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading zict-3.0.0-py2.py3-none-any.whl (43 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m43.3/43.3 kB[0m [31m9.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading colorcet-3.1.0-py3-none-any.whl (260 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m260.3/260.3 kB[0m [31m56.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading linkify_it_py-2.0.3-py3-none-any.whl (19 kB)
    Downloading mdit_py_plugins-0.4.2-py3-none-any.whl (55 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m55.3/55.3 kB[0m [31m14.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading multipledispatch-1.0.0-py3-none-any.whl (12 kB)
    Downloading pyct-0.5.0-py2.py3-none-any.whl (15 kB)
    Downloading xarray-2025.1.2-py3-none-any.whl (1.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m116.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading cloudpickle-3.1.1-py3-none-any.whl (20 kB)
    Downloading importlib_metadata-8.6.1-py3-none-any.whl (26 kB)
    Downloading locket-1.0.0-py2.py3-none-any.whl (4.4 kB)
    Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)
    Downloading msgpack-1.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (403 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m403.7/403.7 kB[0m [31m77.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading partd-1.4.2-py3-none-any.whl (18 kB)
    Downloading sortedcontainers-2.4.0-py2.py3-none-any.whl (29 kB)
    Downloading tblib-3.0.0-py3-none-any.whl (12 kB)
    Downloading uc_micro_py-1.0.3-py3-none-any.whl (6.2 kB)
    Downloading zipp-3.21.0-py3-none-any.whl (9.6 kB)
    Installing collected packages: sortedcontainers, nvtx, nvidia-ml-py, multipledispatch, libkvikio-cu12, cuda-bindings, zipp, zict, xyzservices, uc-micro-py, toolz, tblib, simpervisor, shapely, pyproj, pyogrio, pynvml, pynvjitlink-cu12, param, nvidia-nvjitlink-cu12, nvidia-nvcomp-cu12, nvidia-curand-cu12, nvidia-cublas-cu12, msgpack, mdurl, locket, llvmlite, libucx-cu12, lazy-loader, cuda-python, colorcet, cloudpickle, ucx-py-cu12, treelite, rmm-cu12, pyviz-comms, pyct, partd, nvidia-cusparse-cu12, nvidia-cufft-cu12, numba, markdown-it-py, linkify-it-py, libucxx-cu12, libcudf-cu12, importlib_metadata, cuproj-cu12, xarray, ucxx-cu12, rich, pylibcudf-cu12, nvidia-cusolver-cu12, numba-cuda, mdit-py-plugins, libcuspatial-cu12, geopandas, dask, cucim-cu12, bokeh, panel, libraft-cu12, distributed, datashader, dask-expr, cudf-cu12, rapids-dask-dependency, pylibraft-cu12, libcuvs-cu12, libcugraph-cu12, holoviews, cuspatial-cu12, pylibcugraph-cu12, libcuml-cu12, distributed-ucxx-cu12, dask-cudf-cu12, dask-cuda, cuvs-cu12, raft-dask-cu12, nx-cugraph-cu12, jupyter-server-proxy, cuml-cu12, cugraph-cu12, cuxfilter-cu12
      Attempting uninstall: zipp
        Found existing installation: zipp 1.0.0
        Uninstalling zipp-1.0.0:
          Successfully uninstalled zipp-1.0.0
      Attempting uninstall: lazy-loader
        Found existing installation: lazy_loader 0.3
        Uninstalling lazy_loader-0.3:
          Successfully uninstalled lazy_loader-0.3
      Attempting uninstall: cloudpickle
        Found existing installation: cloudpickle 2.2.1
        Uninstalling cloudpickle-2.2.1:
          Successfully uninstalled cloudpickle-2.2.1
      Attempting uninstall: importlib_metadata
        Found existing installation: importlib-metadata 4.6.4
        Uninstalling importlib-metadata-4.6.4:
          Successfully uninstalled importlib-metadata-4.6.4
    Successfully installed bokeh-3.6.3 cloudpickle-3.1.1 colorcet-3.1.0 cucim-cu12-25.2.0 cuda-bindings-12.8.0 cuda-python-12.8.0 cudf-cu12-25.2.2 cugraph-cu12-25.2.0 cuml-cu12-25.2.1 cuproj-cu12-25.2.0 cuspatial-cu12-25.2.0 cuvs-cu12-25.2.1 cuxfilter-cu12-25.2.0 dask-2024.12.1 dask-cuda-25.2.0 dask-cudf-cu12-25.2.2 dask-expr-1.1.21 datashader-0.17.0 distributed-2024.12.1 distributed-ucxx-cu12-0.42.0 geopandas-1.0.1 holoviews-1.20.1 importlib_metadata-8.6.1 jupyter-server-proxy-4.4.0 lazy-loader-0.4 libcudf-cu12-25.2.2 libcugraph-cu12-25.2.0 libcuml-cu12-25.2.1 libcuspatial-cu12-25.2.0 libcuvs-cu12-25.2.1 libkvikio-cu12-25.2.1 libraft-cu12-25.2.0 libucx-cu12-1.18.0 libucxx-cu12-0.42.0 linkify-it-py-2.0.3 llvmlite-0.43.0 locket-1.0.0 markdown-it-py-3.0.0 mdit-py-plugins-0.4.2 mdurl-0.1.2 msgpack-1.1.0 multipledispatch-1.0.0 numba-0.60.0 numba-cuda-0.2.0 nvidia-cublas-cu12-12.8.4.1 nvidia-cufft-cu12-11.3.3.83 nvidia-curand-cu12-10.3.9.90 nvidia-cusolver-cu12-11.7.3.90 nvidia-cusparse-cu12-12.5.8.93 nvidia-ml-py-12.570.86 nvidia-nvcomp-cu12-4.2.0.11 nvidia-nvjitlink-cu12-12.8.93 nvtx-0.2.11 nx-cugraph-cu12-25.2.0 panel-1.6.1 param-2.2.0 partd-1.4.2 pyct-0.5.0 pylibcudf-cu12-25.2.2 pylibcugraph-cu12-25.2.0 pylibraft-cu12-25.2.0 pynvjitlink-cu12-0.5.2 pynvml-12.0.0 pyogrio-0.10.0 pyproj-3.7.1 pyviz-comms-3.0.4 raft-dask-cu12-25.2.0 rapids-dask-dependency-25.2.0 rich-13.9.4 rmm-cu12-25.2.0 shapely-2.0.7 simpervisor-1.0.0 sortedcontainers-2.4.0 tblib-3.0.0 toolz-1.0.0 treelite-4.4.1 uc-micro-py-1.0.3 ucx-py-cu12-0.42.0 ucxx-cu12-0.42.0 xarray-2025.1.2 xyzservices-2025.1.0 zict-3.0.0 zipp-3.21.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mCollecting gputil
      Downloading GPUtil-1.4.0.tar.gz (5.5 kB)
      Preparing metadata (setup.py) ... [?25ldone
    [?25hBuilding wheels for collected packages: gputil
      Building wheel for gputil (setup.py) ... [?25ldone
    [?25h  Created wheel for gputil: filename=GPUtil-1.4.0-py3-none-any.whl size=7394 sha256=91aa38cbdfbbc1f7b4673189ccfc23140a7006a8e0d4d1766873f5feaa0daa0f
      Stored in directory: /root/.cache/pip/wheels/2b/4d/8f/55fb4f7b9b591891e8d3f72977c4ec6c7763b39c19f0861595
    Successfully built gputil
    Installing collected packages: gputil
    Successfully installed gputil-1.4.0
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m


```python
import warnings

warnings.simplefilter("ignore")

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import cudf
import GPUtil

if torch.cuda.is_available():
    device = "cuda"
else:
    raise

datasets_dir = "../input" if IS_KAGGLE else "../datasets"
kfold = KFold(shuffle=True, random_state=42)
```


```python
fn = "train_poss.csv"
print(f"reading {fn}")
train = pd.read_csv(f"{datasets_dir}/net-dataset/{fn}")

train = pd.concat(
    [
        train.select_dtypes("int64").astype("int32"),
        train.select_dtypes("float64").astype("float32"),
    ],
    axis=1,
)
```

    reading train_poss.csv



```python
print("train")
int_cols = train.select_dtypes("int32").columns.to_list()
print(f"{len(int_cols):>3} {int_cols}")
float_cols = train.select_dtypes("float32").columns.to_list()
o_cols = [c for c in float_cols if c.split("_")[2] == "o"]
print(f"{len(o_cols):>3} {o_cols[:3]} ... {o_cols[-3:]}")
d_cols = [c for c in float_cols if c.split("_")[2] == "d"]
print(f"{len(d_cols):>3} {d_cols[:3]} ... {d_cols[-3:]}")
sos_o_cols = [c for c in float_cols if c.split("_")[3] == "o"]
print(f"{len(sos_o_cols):>3} {sos_o_cols[:3]} ... {sos_o_cols[-3:]}")
sos_d_cols = [c for c in float_cols if c.split("_")[3] == "d"]
print(f"{len(sos_d_cols):>3} {sos_d_cols[:3]} ... {sos_d_cols[-3:]}")
print("---")
print(f"{train.shape[1]} {len(int_cols) + len(o_cols) + len(d_cols) + len(sos_o_cols) + len(sos_d_cols)}")
```

    train
      5 ['Season', 'DayNum', 'TeamID_1', 'TeamID_2', 'Margin']
     28 ['Score_poss_o_1', 'FGM_poss_o_1', 'FGA_poss_o_1'] ... ['Stl_poss_o_2', 'Blk_poss_o_2', 'PF_poss_o_2']
     28 ['Score_poss_d_1', 'FGM_poss_d_1', 'FGA_poss_d_1'] ... ['Stl_poss_d_2', 'Blk_poss_d_2', 'PF_poss_d_2']
     28 ['sos_Score_poss_o_1', 'sos_FGM_poss_o_1', 'sos_FGA_poss_o_1'] ... ['sos_Stl_poss_o_2', 'sos_Blk_poss_o_2', 'sos_PF_poss_o_2']
     28 ['sos_Score_poss_d_1', 'sos_FGM_poss_d_1', 'sos_FGA_poss_d_1'] ... ['sos_Stl_poss_d_2', 'sos_Blk_poss_d_2', 'sos_PF_poss_d_2']
    ---
    117 117



```python
print(f"train: {str(train.shape):>23}")

X_df = train.drop(columns=["Season", "DayNum", "TeamID_1", "TeamID_2", "Margin"])
print(f"X_df: {str(X_df.shape):>24}")

y_s = train["Margin"]
print(f"y_s: {str(y_s.shape):>21}")
scaler_y = StandardScaler()
```

    train:           (202033, 117)
    X_df:            (202033, 112)
    y_s:             (202033,)



```python
def brier_score(y_pred_oof):
    win_prob_pred_oof = 1 / (1 + np.exp(-y_pred_oof * 0.175))
    team_1_won = (y_s > 0).astype("int32")
    return np.mean((win_prob_pred_oof - team_1_won) ** 2)
```


```python
print(f"xgboost")
xgb_models = []
y_pred_oof = np.zeros(y_s.shape[0])

for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X_df.index), 1):
    print(f"  fold {fold_n}")
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
        verbose=500,
        eval_set=[
            (X_fold, y_fold),
            (X_oof, y_oof)
        ],
    )

    xgb_models.append(m)
    y_pred_oof[i_oof] = m.predict(X_oof)
    GPUtil.showUtilization()
    print()

score = brier_score(y_pred_oof)
print(f"xgboost score: {score:.4f}")

for fold_n, m in enumerate(xgb_models, 1):
    fn = f"xgb_{f'{score:.4f}'[2:6]}_{fold_n}.json"
    m.get_booster().save_model(fn)
    print(f"wrote {fn}")
```

    xgboost
      fold 1
    [0]	validation_0-rmse:16.40764	validation_1-rmse:16.48630
    [500]	validation_0-rmse:11.23856	validation_1-rmse:11.36118
    [1000]	validation_0-rmse:10.97439	validation_1-rmse:11.14115
    [1500]	validation_0-rmse:10.89099	validation_1-rmse:11.09311
    [1999]	validation_0-rmse:10.83894	validation_1-rmse:11.07478
    | ID | GPU | MEM |
    ------------------
    |  0 |  0% |  2% |
    
      fold 2
    [0]	validation_0-rmse:16.41837	validation_1-rmse:16.44564
    [500]	validation_0-rmse:11.25720	validation_1-rmse:11.29307
    [1000]	validation_0-rmse:10.99616	validation_1-rmse:11.06826
    [1500]	validation_0-rmse:10.91148	validation_1-rmse:11.02228
    [1999]	validation_0-rmse:10.85869	validation_1-rmse:11.00443
    | ID | GPU | MEM |
    ------------------
    |  0 | 65% |  2% |
    
      fold 3
    [0]	validation_0-rmse:16.42853	validation_1-rmse:16.40116
    [500]	validation_0-rmse:11.24714	validation_1-rmse:11.36190
    [1000]	validation_0-rmse:10.98022	validation_1-rmse:11.13392
    [1500]	validation_0-rmse:10.89689	validation_1-rmse:11.08670
    [1999]	validation_0-rmse:10.84545	validation_1-rmse:11.06761
    | ID | GPU | MEM |
    ------------------
    |  0 | 60% |  2% |
    
      fold 4
    [0]	validation_0-rmse:16.44140	validation_1-rmse:16.35334
    [500]	validation_0-rmse:11.24083	validation_1-rmse:11.31784
    [1000]	validation_0-rmse:10.98137	validation_1-rmse:11.11179
    [1500]	validation_0-rmse:10.89875	validation_1-rmse:11.06529
    [1999]	validation_0-rmse:10.84727	validation_1-rmse:11.04490
    | ID | GPU | MEM |
    ------------------
    |  0 | 38% |  2% |
    
      fold 5
    [0]	validation_0-rmse:16.42101	validation_1-rmse:16.43289
    [500]	validation_0-rmse:11.25059	validation_1-rmse:11.33162
    [1000]	validation_0-rmse:10.98508	validation_1-rmse:11.10273
    [1500]	validation_0-rmse:10.90304	validation_1-rmse:11.05417
    [1999]	validation_0-rmse:10.85003	validation_1-rmse:11.03566
    | ID | GPU | MEM |
    ------------------
    |  0 | 60% |  2% |
    
    xgboost score: 0.1609
    wrote xgb_1609_1.json
    wrote xgb_1609_2.json
    wrote xgb_1609_3.json
    wrote xgb_1609_4.json
    wrote xgb_1609_5.json



```python
print("torch")

X = torch.as_tensor(
    StandardScaler().fit_transform(X_df.values),
    dtype=torch.float32,
    device=device,
)
print(f"X:    {X.shape}")

y = torch.tensor(
    scaler_y.fit_transform(train[["Margin"]]).flatten(),
    dtype=torch.float32,
    device=device,
)
print(f"y:    {y.shape}")

def weight(*size):
    return torch.nn.Parameter(0.1 * torch.randn(*size, device=device))

def bias(*size):
    return torch.nn.Parameter(torch.zeros(*size, device=device))

mse_ = torch.nn.MSELoss()

def mse(y_pred_epoch, i):
    return  mse_(y_pred_epoch, y[i].view(-1, 1))

def aslist(param):
    return param.cpu().detach().numpy().tolist()

def aspy(m):
    return {
        "w": [aslist(w) for w in m["w"]],
        "b": [aslist(b) for b in m["b"]],
    }

n_epochs = 1_000
hidden_size = 64
torch_models = []

y_pred_oof = torch.zeros(
    y.shape[0],
    dtype=torch.float32,
    requires_grad=False,
    device=device,
)

for fold_n, (i_fold, i_oof) in enumerate(kfold.split(X_df.index), 1):
    print(f"  fold {fold_n}")

    m = {
        "w": [
            weight(X_df.shape[1], hidden_size),
            weight(hidden_size, 1),
        ],
        "b": [
            bias(hidden_size),
            bias(1),
        ],
    }

    optimizer = torch.optim.Adam(m["w"] + m["b"], weight_decay=1e-4)

    def forward(i):
        return F.leaky_relu(X[i] @ m["w"][0] + m["b"][0], negative_slope=0.1) @ m["w"][1] + m["b"][1]

    for epoch_n in range(1, n_epochs + 1):
        y_pred_epoch_fold = forward(i_fold)
        mse_epoch_fold = mse(y_pred_epoch_fold, i_fold)
        optimizer.zero_grad()
        mse_epoch_fold.backward()
        optimizer.step()

        if (epoch_n % (n_epochs // 2) == 0) or (epoch_n > (n_epochs - 3)):
            with torch.no_grad():
                y_pred_epoch_oof = forward(i_oof)
                mse_epoch_oof = mse(y_pred_epoch_oof, i_oof)

            print(
                f"    epoch {epoch_n:>6}: "
                f"fold={mse_epoch_fold.item():.4f} "
                f"oof={mse_epoch_oof.item():.4f}"
            )

    with torch.no_grad():
        y_pred_oof[i_oof] = forward(i_oof).flatten()

    GPUtil.showUtilization()
    torch_models.append(aspy(m))
    print()

y_pred_oof = scaler_y.inverse_transform(
    y_pred_oof.cpu().numpy().reshape(-1, 1)
).flatten()

score = brier_score(y_pred_oof)
print(f"torch score:   {score:.4f}")

for fold_n, m in enumerate(torch_models, 1):
    fn = f"nn_{f'{score:.4f}'[2:6]}_{fold_n}.json"
    with open(fn, 'w') as f:
        json.dump(m, f)
    print(f"wrote {fn}")
```

    torch
    X:    torch.Size([202033, 112])
    y:    torch.Size([202033])
      fold 1
        epoch    500: fold=0.4414 oof=0.4509
        epoch    998: fold=0.4353 oof=0.4479
        epoch    999: fold=0.4353 oof=0.4479
        epoch   1000: fold=0.4352 oof=0.4479
    | ID | GPU | MEM |
    ------------------
    |  0 | 88% |  4% |
    
      fold 2
        epoch    500: fold=0.4406 oof=0.4416
        epoch    998: fold=0.4348 oof=0.4403
        epoch    999: fold=0.4347 oof=0.4402
        epoch   1000: fold=0.4347 oof=0.4403
    | ID | GPU | MEM |
    ------------------
    |  0 | 89% |  4% |
    
      fold 3



```python

```
