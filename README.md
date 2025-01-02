# Cell Segmentation and Trace Intensity Plotter
The following is a project that leverages Python, CV2 and NumPy to perform cell segmentation and trace intensity plotting on a given dataset. Included are both a pure cpu-written script as well as a GPU-enabled version that leverages CUDA via CuPy. This repository is visible for archival purposes and is not being worked on.


## Installation Instructions
### Prerequisites
You will need a Python interpreter to run this application, preferably `python3`. If you wish to run the GPU-accelerated version of the script, you must also have a valid Nvidia GPU installed that supports CUDA.

### Regular CPU version
Run `requirements-cpu.txt` by invoking:
```bash
pip install -r requirements-cpu.txt
```

Then, run `cpu-script.py` on the dataset by invoking:
```
python cpu-script.py
```

### CUDA GPU Version
First verify whether you have CUDA installed.
```bash
nvcc --version
```
Install CUDA if needed using a package manager or other method.

Next, run `requirements-gpu.txt` by invoking:
```bash
pip install -r requirements-gpu.txt
```

Then, run `gpu-script.py` on the dataset by invoking:
```
python gpu-script.py
```

### Dataset Path
Move any `.avi` dataset into a newly created `data` directory. The program expects it to be called
`dataset.avi`, but if you want a different name go into the script and change the `DATA_PATH` constant
to the name of the file.

### Batch Size (GPU)
If you run into out of memory errors, adjust the batch size in the script's `BATCH_SIZE` constant.
100 works for 10GB VRAM, but you might have to adjust this down if you have less.


## Usage
Drop any `.avi` format dataset into the `data` folder and run whichever version you like. Printed to the console will be the number of ROIs detected as well as the total time needed to perform the operation. Once the script has finished execution, the plots should be visible.

If you would like a sample plot of this program on a given dataset, please check the `sample_output` folder.

