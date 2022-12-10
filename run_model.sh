#!/usr/bin/env bash

# Preprocess Data
python model_ecaas_agrifieldnet_bronze/prepare_data.py

# Inference
python model_ecaas_agrifieldnet_bronze/run_boosting_models.py



