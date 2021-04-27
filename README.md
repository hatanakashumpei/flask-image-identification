# Flask Image Identification

## setup

```#bin/bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Trouble Sheetings

1. **Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR**
Running the following command solved the problem successfully.

```#bin/bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
```
