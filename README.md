# KatanAI

KatanAI is a Python project using Deep Learning technologies to segment the right 
ventricle in cardiac cine-MRI images.

## Motivations 

Sate-of-the-art models used in cardiac images segmentation present good performances, however, their generalizability 
capacities remain problematics. Due to high variance in cardiac images datasets, models performances tend to drop 
significantly when tested on a different dataset than the one used for training. This project aims to analyse the 
generalisability capacities of two models (Unet and FCN-8) by realising cross-training-testing experiments on two 
cardiac cine-MRI datasets (Automated Cardiac Diagnosis Challenge (ACDC), and Multi-Disease, Multi-View & Multi-Center 
Right Ventricular Segmentation in Cardiac MRI (M&Ms-2)). A third dataset (Oxford pets III) has been used as sample.  
Both cine-MRI datasets firstly underwent multi-step pre-processing transformations. The segmentation task was focused 
on the right ventricle (RV). Cross-training-testing experiments presented the expected drop in performances, exception 
made of Unet when trained on ACDC and tested on M&Ms-2. The model performed better in this situation than when trained 
on M&Ms-2 itself. The reverse experiment presented a drop in performance but very moderately. This finding leads to 
believe that Unet could be a good architecture choice for potential clinical deployment. Further comparisons and 
experiments still need to be done.

## Data
Place the data.zip directory provided in tha additional documents in the root of the project (./KatanAI/) 

Unzip the directory with the following command

```bash
unzip data.zip
```

This directory contains for each dataset:
- The raw directory (as downloaded from the official source) (name: raw).
- Two raw directories representing the split between the testing and training sets (names: training, testing). 
- Two directories with normalized images (names: testing_normalized, training_normalized).

Re-running the pre-processing pipelines will overwrite the splited and normalized directories. 

## Trained models

Place the saved_model.zip file provided in the additional documents in the root of the project (./KatanAI/)

Unzip the directory with the following command

```bash
unzip saved_model.zip
```

Every new training will overwrites the corresponding saved model.

## Python Virtual Environment Installation
Using Python 3.9.6 and pip 21.1.3 

Execute the following commands in order:
```bash
pip -m venv ./venv
```

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```
python main.py
```

## Docker Installation
Using Docker version 20.10.7

Docker Deamon need to be up and running. 

Execute the following commands in order: 
```bash
sudo docker build -t katanai .
```

```bash
docker run katanai
```
