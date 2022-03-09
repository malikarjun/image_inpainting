Image Inpainting

Run
```
python dataset_prep.py
```
This script downloads, extracts and stores a subset of imagenet into the `data` directory within the current working 
directory.

Folder structure
```
-data
    -inpaint
        -original
            -train
            -train_mask
            -val
            -val_mask
        -line_mask
            -train
            -val
```

`original` folder contains the uncorrupted images, `line_mask` contains corrupted images which have random noise in the 
form lines with varying thickness, and `random_mask` contains random noise across all pixels. 


`train_mask` contains `.npy` files which are 2D numpy arrays where the value at `(i, j)`th location is 1 if that value
is known in the image, 0 otherwise.

`train` - 9296 images
`val` - 3856 images


