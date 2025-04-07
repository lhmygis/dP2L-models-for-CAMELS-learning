# dP2L-models-for-CAMELS-learning：
dP2L was developed by Heng Li, Chunxiao Zhang* and others from China University of Geosciences (Beijing) for rainfall-runoff learning in large sample basins. 
The related manuscript is currently under peer review in Journal of Hydrology.

# Model dependency library：
The python environment required to run the GPU version dP2L model includes: absl-py 0.15.0 pypi_0 pypi astunparse 1.6.3 pypi_0 pypi bzip2 1.0.8 h2466b09_7 conda-forge ca-certificates 2024.7.4 h56e8100_0 conda-forge clang 5.0 pypi_0 pypi flatbuffers 1.12 pypi_0 pypi gast 0.4.0 pypi_0 pypi google-pasta 0.2.0 pypi_0 pypi grpcio 1.65.4 pypi_0 pypi h5py 3.1.0 pypi_0 pypi importlib-metadata 8.2.0 pypi_0 pypi keras 2.6.0 pypi_0 pypi keras-preprocessing 1.1.2 pypi_0 pypi libffi 3.4.2 h8ffe710_5 conda-forge libsqlite 3.46.0 h2466b09_0 conda-forge libzlib 1.3.1 h2466b09_1 conda-forge markdown 3.6 pypi_0 pypi markupsafe 2.1.5 pypi_0 pypi numpy 1.19.5 pypi_0 pypi openssl 3.3.1 h2466b09_2 conda-forge opt-einsum 3.3.0 pypi_0 pypi pandas 1.2.4 pypi_0 pypi pip 24.2 pyhd8ed1ab_0 conda-forge protobuf 3.20.0 pypi_0 pypi python 3.9.19 h4de0772_0_cpython conda-forge python-dateutil 2.9.0.post0 pypi_0 pypi pytz 2024.1 pypi_0 pypi scipy 1.7.3 pypi_0 pypi setuptools 72.1.0 pyhd8ed1ab_0 conda-forge six 1.15.0 pypi_0 pypi tensorboard 2.17.0 pypi_0 pypi tensorboard-data-server 0.7.2 pypi_0 pypi tensorflow-estimator 2.15.0 pypi_0 pypi tensorflow-gpu 2.6.0 pypi_0 pypi termcolor 1.1.0 pypi_0 pypi tk 8.6.13 h5226925_1 conda-forge typing-extensions 3.7.4.3 pypi_0 pypi tzdata 2024a h0c530f3_0 conda-forge ucrt 10.0.22621.0 h57928b3_0 conda-forge vc 14.3 h8a93ad2_20 conda-forge vc14_runtime 14.40.33810 ha82c5b3_20 conda-forge vs2015_runtime 14.40.33810 h3bf8584_20 conda-forge werkzeug 3.0.3 pypi_0 pypi wheel 0.43.0 pyhd8ed1ab_1 conda-forge wrapt 1.12.1 pypi_0 pypi xz 5.2.6 h8d14728_0 conda-forge zipp 3.19.2 pypi_0 pypi

# Path and file comments：
"../the project path/"：Model project root directory  
"../the project path/basin_list.txt"：List text of basin ids  
"../the project path/dP2L_class.py"：Model class file  
"../the project path/dP2L1_Train_Demo.py"：Model training file  
"../the project path/dataprocess.py"：CAMELS data preprocessing file  
"../the project path/loss.py"：Model loss function file  
  
"../the project path/CAMELS_attributes/attributedata_671.csv"：CAMELS basin attribute data (unstandardized)  
"../the project path/camels_h5/usgs_streamflow/HUC_id(e.g., 01, 02, 03, 04, 05...)/{basin_id}_streamflow_qc.txt"：CAMELS basin streamflow data  
"../the project path/camels_h5/basin_mean_forcing/daymet/HUC_id(e.g., 01, 02, 03, 04, 05...)/{basin_id}_lump_cida_forcing_leap.txt"：CAMELS basin mean_forcings

"../the project path/Models_h5/"：Model storage path  


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Quick Start：
The code was tested with Python 3.6. To use this code, please do:

1. Clone the repo using Conda. Note*：In the tensorflow_2.6.yml file, the last line `prefix: ..\Users\anaconda3\envs\tensorflow_2.6` specifies the installation path of the Conda environment.
   Please make sure to change this path to match the location of your own Conda installation. This line is automatically generated when exporting environments, and keeping an incorrect path may cause issues during environment creation:

   ```shell
   conda env create -f tensorflow_2.6.yml -n dP2L_environment
   ```

2. Please change the following working path (dP2L1_Train_Demo.py) to your owns:

   ```shell
   working_path = "../the project path"
   attrs_path = "../the project path/CAMELS_attributes/attributedata_671.csv" 
   ```
