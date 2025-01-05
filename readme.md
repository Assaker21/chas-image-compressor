# .chas image compression

This is an image compressing algorithm based on Variable Block Sizes and Run-Length Encoding, developed for the Multimedia Course, ULFG2.

## What's all of these files ?

The `main.py` is responsible for compressing and decompressing the input.png file, and outputting:

1. `input.raw`: The input in .raw format
2. `output.chas`: The final compressed image in .chas format.
3. `output.png`: The final compressed image in .png format.

The `main_multiple.py` is responsible for compressing `input.png` into multiple output files, with different hyperparameters.

The parameters that change are the minimum block size and the MSE threshold.

## How to use

1. Open the directory of the .py files
2. Run the command `python <filename>.py`, by replacing `<filename>` by `main_multiple` or `main`, depeding on your needs.
