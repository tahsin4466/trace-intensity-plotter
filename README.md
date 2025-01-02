## Installation Instructions
## Regular CPU version
Run `requirements-cpu.txt` by invoking:
```bash
pip install -r requirements-cpu.txt
```

Then, run `cpu-script.py` on the dataset.

## CUDA GPU Version
First verify whether you have CUDA installed. You need an Nvidia GPU to run CUDA:
```bash
nvcc --version
```
Install CUDA if needed using a package manager or other method.

Next, run `requirements-gpu.txt` by invoking:
```bash
pip install -r requirements-gpu.txt
```

Then, run `gpu-script.py` on the dataset.


## Setting Constants
### Dataset Path
Move the `.avi` dataset into the `data` folder. The program expects it to be called
`dataset.avi`, but if you want a different name go into the script and change the `DATA_PATH` constant
to the name of the file.

### Batch Size (GPU)
If you run into out of memory errors, adjust the batch size in the script's `BATCH_SIZE` constant.
100 works for 10GB VRAM, but you might have to adjust this down if you have less.
