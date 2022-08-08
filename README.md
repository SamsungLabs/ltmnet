# Learning Tone Curves for Local Image Enhancement
Code and dataset generation scripts for article: [Learning Tone Curves for Local Image Enhancement](https://ieeexplore.ieee.org/document/9784427).

Luxi Zhao, Abdelrahman Abdelhamed, Michael S. Brown

Samsung Artificial Intelligence Center, Toronto, Canada
## Experiments on HDR+ Dataset
### Dataset Preparation
#### Generating HDR+ input / GT pairs
1. Download [HDR+ dataset](https://console.cloud.google.com/storage/browser/hdrplusdata/20171106) 

2. Run the following command to process the raw input (`20171106/results_20171023/<burst_name>/merged.dng`) up to the gamma-correction stage. `20171106/results_20171023/<burst_name>/final.jpg` is used as ground truth

    ```
    python3 -m prepare.prep_hdrplus --hdrplus_dir /path/to/20171106 --out_dir <dataset_dir>
    mv <dataset_dir>/gt-final <dataset_dir>/gt
    mv <dataset_dir>/input-srgb-gamma <dataset_dir>/input
    ```

#### Train / Valid / Test split
- Training data file names: [prepare/data/hdrplus_images_train.txt](prepare/data/hdrplus_images_train.txt) 
- Validation data file names: [prepare/data/hdrplus_images_val.txt](prepare/data/hdrplus_images_val.txt) 
- Testing data file names: [prepare/data/hdrplus_images_test.txt](prepare/data/hdrplus_images_test.txt) 
### Training
#### Training LTMNet on HDR+ data
```
python3 -m jobs.ltmnet_hdrplus_ds
```

#### Training LTMNet with residual module on HDR+ data
```
python3 -m jobs.ltmnet_res_hdrplus_ds
```

Evaluation results are saved to `<project_root>/outputs`.

## Experiments on Our LTM Dataset
### Dataset Preparation
Download [MIT-Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/).

Input:
1. Export MIT-Adobe FiveK images from LightRoom with the following settings
    - Collection: Input/InputZeroed with ExpertC WhiteBalance
    - Filetype: PNG
    - Resize: long edge resized to 1024 pixels, 240 ppi
    - Bit depth: 8 bit
    - Color Space: sRGB
2. Save the exported images to `<dataset_dir>/input`

Ground truth:

```
python3 -m jobs.job_prep_mit_adobe_clahe_ds
cp -r <dataset_dir>/mit-adobe-clahe-15v/long-edge-1024 <dataset_dir>/gt
```

### Training LTMNet on our LTM dataset
```
python3 -m jobs.ltmnet_ltm_ds
```
Evaluation results are saved to `<project_root>/outputs`.

## Experiments on MIT-Adobe FiveK Dataset
### Dataset Preparation
Download [MIT-Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/).

Input:
1. Export MIT-Adobe FiveK images from LightRoom with the following settings
- Collection: Input/InputZeroed with ExpertC WhiteBalance
- Filetype: PNG
- Resize: long edge resized to 1024 pixels, 240 ppi
- Bit depth: 8 bit
- Color Space: sRGB
2. Save the exported images to `<dataset_dir>/input`

Ground truth:
1. Export MIT-Adobe FiveK images from LightRoom with the following settings
- Collection: Experts/C
- Filetype: PNG
- Resize: long edge resized to 1024 pixels, 240 ppi
- Bit depth: 8 bit
- Color Space: sRGB
2. Save the exported images to `<dataset_dir>/gt`

Train / Valid / Test split
```
python3 -m prepare.gen_file_lists \
    --out_dir "~/Data" \
    --train_range 101 1100 \
    --val_range 1 100 \
    --test_range 4501 5000
```
- Train indices: a0102 - a1101
- Validation indices: a0002 - a0101
- Test indices: a4502 - a5000

### Training
Coming soon ...

## Pretrained Models
LTMNet trained on HDR+ dataset: [pretrained_models/ltmnet_hdrplus_ds_model](pretrained_models/ltmnet_hdrplus_ds_model)

LTMNet with residual module trained on HDR+ dataset: [pretrained_models/ltmnet_res_hdrplus_ds_model](pretrained_models/ltmnet_res_hdrplus_ds_model)

LTMNet trained on our LTM dataset: [pretrained_models/ltmnet_ltm_ds_model](pretrained_models/ltmnet_ltm_ds_model)
### To run a pretrained model
Use the following arguments for `main.py`
```
--pretrained_model_dir ./pretrained_models/ltmnet_hdrplus_ds_model
--eval
```


> Note: For now the code only supports bit_depth = 8.