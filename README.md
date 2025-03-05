# 1DT116\_Assignment
Assignment of 1DT116 Low-level Parallel Programming at Uppsala University

## Preparing to run on your machine

(Jump over to the building section if you will be using the machine that we provided)

We will provide instructions to run the codebase on an Ubuntu system.
If you use Windows, you should be able to use WSL2 to achieve the same goal.

### Prerequisite: Qt5 (optional)

If you would rather not use `qt-mode` and instead only use the `export-trace`
mode, then you can skip this step. The configure step will take care of this.

Qt5 is optionally required to build and run the assignment using qt-mode. Tested with Qt5 on Ubuntu
24.04.1. Installed Qt5 with the following command: (You'll probably need to run with sudo)

```
# apt install qtbase5-dev
```

If you cannot install Qt5, but you do have Qt4 then you will need to modify
the include and library arguments in `demo/Makefile`. Change the `qt5` strings
under `QTINCLUDES` and `LIBS` to your Qt version.


### Prerequisite: TinyXML-2

The configuration files of this project are specified in XML. We rely on the TinyXML2 library to read in the configuration files.
If you are on Ubuntu, the following command should install the library for you.

```
# apt install libtinyxml2-dev
```

### Prerequisite: CUDA

The 2nd (optional) and 4th part of the assignment requires running the code using CUDA. 
This also means that you need to have a Nvidia GPU that is capable of running CUDA.

If this is the case, please go ahead to the [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and download and install the CUDA toolkit and the [CUDA drivers](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html).
I used the [Network Repo Installation for Ubuntu](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-ubuntu) approach then followed the common installation instructions (next section that follows). Be sure to read and follow the pre-installation steps.

Make sure to add the CUDA `bin` directory to the PATH. 

I added the following line to a new file called `/etc/profile.d/cuda.sh` (need to use sudo to make this file. Note that your binaries may be located in a different directory).

```
export PATH=$PATH:/usr/local/cuda/bin
```

Also, you need to change the permission of the file.

```
sudo chmod 0644 /etc/profile.d/cuda.sh
```

Try running any of the following commands to check if CUDA is setup properly.

```
$ nvidia-smi # This command should list your GPU
$ nvcc --version # This command should print the version of your cuda compiler
```

You could also check the [post-installation steps](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#post-installation-actions), which include more detailed instructions on setting the PATH, and try verifying whether our install was successful by running some sample code.

## Configure your machine
Whether you are working on MacOS or on Ubuntu, if you have CUDA and/or QT
installed on your Ubuntu machine, the makefile will need to slightly differ.

The provided `configure.sh` file should take care of this for you. If you are
not using Ubuntu, you may need to slightly modify the `is_ubuntu()` function
in the script to allow your system to be recognized. (Ask ChatGPT for help).

```
$ ./configure.sh
```

This step is required whenever you first clone your repository. (Or if you
install the CUDA framework, or the QT5 framework, you'll need to configure
again.)


## Building
Now once you have setup your machine OR you are running on our provided machine, you can start building and running the project.

```
$ make
```

This should build both binaries in `libpedsim` then in `demo`

### If you are using your own machine

If during the build the compiler complains of unknown path to the Qt headers
(include path), check the path in `QTINCLUDES` and make sure they exist on
your system. Your system may put the include files at a different location.
The makefile currently tries to locate the Qt header files using the following
command:
```
$ qmake -query QT_INSTALL_HEADERS
```

If this fails, try to locate the header files using the following command:
```
$ find / -name QTHEADER.h` (where the QTHEADER.h the compiler is looking for)
```
Using the results fix the path in `QT_HEADERS` variable in the `demo/Makefile`

If working with Ubuntu and you use the command above to install QT, this shouldn't be an issue... (Let us know if it is.)


## Running
If the build is successful, run the simulator using the following command

```
$ demo/demo scenario.xml
```

If you get an error about not finding any `display`, then you need to connect
to the provides server with an X session.

On Mac/Linux, on your terminal, add the `-Y` argument in your `ssh` command.
On Windows, start your X-server application (e.g. Xming), and configure
enabling X11 in your PuTTY configuration.
