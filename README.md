# **VAIPE_classification**
This repository implements the classification module in VAIPE 2022 competion.
## **How to train**

### Prepare data
```
python convert_to_csv.py  #root_dir containts images and labels
python preprocesing_data.py
```

### Training
Modify hyperparameters at **config.py** and run command
```
python train.py
```

