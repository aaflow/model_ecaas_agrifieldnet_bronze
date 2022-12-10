# 3rd place solution of Zindi AgriFieldNet India Challenge


## Ensemble of Boostings models and Neural Networks for crop types classification 

Inference of LightGBM, XGBoost models, trained on aggregation field features and Neural Network models with Encoder to process raw field images and Dense layers to process numerical features.

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
        * `input/ref_agrifieldnet_competition_v1`: data for inferencing.
        * `input/checkpoint`: the model checkpoint.
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
