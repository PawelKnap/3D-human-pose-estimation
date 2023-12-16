USER GUIDE
==========================

## Contents
1. [Operating System, requirements & radar hardware](#operating-system,-requirements-&-radar-hardware)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Run](#run) 

## Operating System, requirements & radar hardware
- Operating System
    - **Windows 10**.
    - **Windows 11**.

- Requirements
    - CUDA (Nvidia GPU) version:
        - NVIDIA graphics card with at least 8 GB available.
        - At least 3 GB of free RAM memory
    - Highly recommended: a CPU with at least 8 cores.

- Advanced tip: You might need more resources with greater `--net_resolution` and/or `scale_number` or less resources by reducing them.

- Radar hardware
    - 3 Texas Instrument (TI) AWR1843BOOST radar boards.

## Prerequisites
1. Install **CMake GUI**: Download and install the `Latest Release` of CMake `Windows win64-x64 Installer` from the [CMake download website](https://cmake.org/download/), called `cmake-X.X.X-win64-x64.msi`.

2. Install **Microsoft Visual Studio (VS) 2019 Enterprise**:
    - **IMPORTANT**: Enable all C++-related flags when selecting the components to install.

3. Nvidia GPU version prerequisites:
    1. Upgrade your Nvidia drivers to the latest version (in the Nvidia "GeForce Experience" software or its [website](https://www.nvidia.com/Download/index.aspx)).
    2. Install [**CUDA 11.1.1**](https://developer.nvidia.com/cuda-11.1.1-download-archive):
        - Install CUDA 11.1.1 after Visual Studio 2019 Enterprise is installed to assure that the CUDA installation will generate all necessary files for VS. If CUDA was installed before installing VS, then re-install CUDA.
        - **Important installation tips**:
            - If CMake returns and error message similar to `CUDA_TOOLKIT_ROOT_DIR not found or specified` or any other CUDA component missing, then: 1) Re-install Visual Studio 2019 Enterprise; 2) Reboot your PC; 3) Re-install CUDA (in this order!).

    3. Install [**cuDNN 8.1.0**](https://developer.nvidia.com/cudnn):
        - In order to manually install it, just unzip it and copy (merge) the contents on the CUDA folder, usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{version}` in Windows.

4. **Caffe, OpenCV, and Caffe prerequisites**:
    - CMake automatically downloads all the Windows DLLs. Alternatively, you might prefer to download them manually:
        - Dependencies:
            - Note: Leave the zip files in `3rdparty/windows/` so that CMake does not try to download them again.
            - Caffe (if you are not sure which one you need, download the default one):
                - [CUDA Caffe (Default)](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe_16_2020_11_14.zip): Unzip as `3rdparty/windows/caffe/`.
                - [CPU Caffe](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe_cpu_2018_05_27.zip): Unzip as `3rdparty/windows/caffe_cpu/`.
                - [OpenCL Caffe](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe_opencl_2018_02_13.zip): Unzip as `3rdparty/windows/caffe_opencl/`.
            - [Caffe dependencies](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe3rdparty_16_2020_11_14.zip): Unzip as `3rdparty/windows/caffe3rdparty/`.
            - [OpenCV 4.2.0](http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/opencv_450_v15_2020_11_18.zip): Unzip as `3rdparty/windows/opencv/`.

5. Install [Python 3.6.8](https://www.python.org/downloads/release/python-368/) version for Windows, and then:
```
sudo pip install numpy opencv-python
```
- **IMPOTANT**: make sure Python 3.6.8 is the Python version of machine as well as anaconda environment.

6. Radars:
    - Download and install the Texas Instruments [MMWAVE-SDK](https://www.ti.com/tool/MMWAVE-SDK), which includes the XDS110 drivers necessary. 
    - Dowload and install the TI tool called [Uniflash](https://www.ti.com/tool/UNIFLASH). It flashes the .bin file to the board. 
    - The board needs a power supply - recommended 5V dc output, 3A output, and a USB connection the PC (this cable should come with the radar)

## Installation

### Clone this project
Download project.zip (https://drive.google.com/file/d/1prmICIxBDwJRRYW0OYJWWYGn2T6Ayo0d/view?usp=sharing) and unzip it.

### People counting algorithm flashing
If the radars are not programmed:
1. Download and unzip the `lab0011-pplcount` folder
2. Open the Uniflash application
3. Set the jumpers on the board to position ‘101’. 
4. Press the SW2 switch (nRST) to power cycle the board. 
5. In the Uniflash application, when prompted to upload a file, navigate to the `lab0011-pplcount` folder, then into the `lab0011_pplcount_quickstart` folder and upload the file `xwr16xx_pcount_lab.bin`. Then follow any other instructions in the Uniflash application.
6. Once you are told that the board is flashed in the Uniflash application, unplug the power from the board
7. Set the jumpers in functional mode – ‘001’
8. Power up the board again, and then power cycle (using nRST). 

### CMake Configuration
1. Go to >> DMake >> bin and open cmake-gui.exe
2. In the "Where is the source code:" tab enter a path to the dowloaded and unzipped project.zip folder
3. In the "Where to build binaries:" tab enter the same path as above but with "\build" at the end. If `build` does not exist, it will ask you whether to create it. Press `Yes`.
<p align="center">
    <img src="Pictures from User Guide/cmake_im_1.png" width="480">
</p>
4. Press the `Configure` button, set the generator to Visual Studio 19 Enterprise, and press `Finish`. Note: CMake-GUI has changed their design after version 14. For versions older than 14, you usually select `Visual Studio XX 20XX Win64` as the generator (`X` depends on your VS version), while the `Optional toolset to use` must be empty. However, new CMake versions require you to select only the VS version as the generator, e.g., `Visual Studio 16 2019`, and then you must manually choose `x64` for the `Optional platform for generator`. See the following images as example.
<p align="center">
    <img src="Pictures from User Guide/cmake_im_2.png" width="240">
    <img src="Pictures from User Guide/cmake_im_2_windows.png" width="240">
    <img src="Pictures from User Guide/cmake_im_2_windows_new.png" width="240">
</p>

5. Enable the `BUILD_PYTHON` flag and click `Configure` again.
6. If this step is successful, the `Configuring done` text will appear in the bottom box in the last line. Otherwise, some red text will appear in that same bottom box.
<p align="center">
    <img src="Pictures from User Guide/cmake_im_3.png" width="480">
    <img src="Pictures from User Guide/cmake_im_3_windows.png" width="480">
</p>

7. Press the `Generate` button and proceed to [Compilation](#compilation). You can now close CMake.


### Compilation

1. Open Visual Studio 19 Enterprise by clicking in `Open Project` in CMake (or alternatively `build/OpenPose.sln`). Then, set the configuration from `Debug` to `Release`.
2. Press <kbd>F7</kbd> (or `Build` menu and click on `Build Solution`).
3. **Important**: Make sure not to skip step 2, it is not enough to click on <kbd>F5</kbd> (Run), you must also `Build Solution` for the Python bindings to be generated.
4. After it has compiled, and if you have a webcam, you can press the green triangle icon (alternatively <kbd>F5</kbd>) to run the OpenPose demo with the default settings on the webcam.

**VERY IMPORTANT NOTE**: In order to use the project outside Visual Studio, and assuming you have not unchecked the `BUILD_BIN_FOLDER` flag in CMake, copy all DLLs from `{build_directory}/bin` into the folder where the generated `openpose.dll` and `*.exe` demos are, e.g., `{build_directory}x64/Release`.

### CONDA environments
Go into the "project" folder and create two conda environments:
```
conda env create -f 3dpose.yml
conda env create -f visualiser.yml
```

The project folder can be found in 

## Run
1. Change the camera number in the Python program according to the order of the cameras in the device manager
2. Change paths in the main() of that program according to the locations of the visualiser environment on your machine
3. Go into the "project" folder and run the program:
```
conda activate 3dpose
python poseestimation.py
```
The "project" folder is -> openpose -> build -> examples -> tutorial_api_python
## Update
For the updated version of the program with 4 radars instead of 3 and improved pose estimation use the pose_estimation_v2.py instead of pose_estimation.py. To do that, the updated code has to be moved to the "project" folder.
