# 3rd place solution of Zindi AgriFieldNet India Challenge


## Ensemble of Boostings models and Neural Networks for crop types classification 

{{ Model Description (paragraph) }}

![{{model_id}}](https://radiantmlhub.blob.core.windows.net/frontend-dataset-images/odk_sample_agricultural_dataset.png)

MLHub model id: `model_ecaas_agrifieldnet_bronze_v1`. Browse on [Radiant MLHub](https://mlhub.earth/model/model_ecaas_agrifieldnet_bronze_v1).

## ML Model Documentation

Please review the model architecture, license, applicable spatial and temporal extents
and other details in the [model documentation](/docs/index.md).

## System Requirements

* Git client
* [Python 3.8](https://www.python.org)

## Hardware Requirements

|Inferencing|Training|
|-----------|--------|
|32 GB RAM | 32 GB RAM|

## Get Started With Inferencing

First clone this Git repository.

```bash
git clone https://github.com/aaflow/model_ecaas_agrifieldnet_bronze.git
cd model_ecaas_agrifieldnet_bronze/
```

After cloning the model repository, install Python dependencies locally in your environment.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```


## Run Model to Generate New Inferences


1. Prepare your input and output data folders. The `data/` folder in this repository
    contains some placeholder files to guide you.

    * The `data/` folder must contain:
        * `input/chips` {{ Landsat, Maxar Open-Data 30cm, Sentinel-2, etc. }} imagery chips for inferencing:
            * File name: {{ `chip_id.tif` }} e.g. {{ `0fec2d30-882a-4d1d-a7af-89dac0198327.tif` }}
            * File Format: {{ GeoTIFF, 256x256 }}
            * Coordinate Reference System: {{ WGS84, EPSG:4326 }}
            * Bands: {{ 3 bands per file:
                * Band 1 Type=Byte, ColorInterp=Red
                * Band 2 Type=Byte, ColorInterp=Green
                * Band 3 Type=Byte, ColorInterp=Blue
                }}
        * `/input/checkpoint` the model checkpoint {{ file | folder }}, `{{ checkpoint file or folder name }}`.
            Please note: the model checkpoint is included in this repository.
    * The `output/` folder is where the model will write inferencing results.

2. Set `INPUT_DATA` and `OUTPUT_DATA` environment variables corresponding with
    your input and output folders. These commands will vary depending on operating
    system and command-line shell:

    ```bash
    # change paths to your actual input and output folders
    export INPUT_DATA="/home/my_user/model_ecaas_agrifieldnet_bronze/data/input/"
    export OUTPUT_DATA="/home/my_user/model_ecaas_agrifieldnet_bronze/data/output/"
    ```

3. Run the `run_model.sh` bash shell script.

    ```bash
    bash run_model.sh
    ```

4. Wait for the script to finish running, then inspect the
`OUTPUT_DATA` folder for results.


## Understanding Output Data

Please review the model output format and other technical details in the [model
documentation](/docs/index.md).
