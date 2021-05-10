# Flask Image Identification

## setup

```#bin/bash
pip install --upgrade pip
pip install -r requirements.txt
```

## How to Use

```#bin/bash
export FLASK_APP=predictfile.py
export FLASK_ENV=development
python -m flask run --host=0.0.0.0 --port=8080 --without-threads
```

## Trouble Sheetings

1. **Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR**<br>
Running the following command solved the problem successfully.

```#bin/bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
```
