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
            -val
        -line_mask
            -train
            -val
```

`original` folder contains the uncorrupted images, and `line_mask` contains corrupted images which have random noise 
in the form lines associated with them.

`train` - 9296 images
`val` - 3856 images


