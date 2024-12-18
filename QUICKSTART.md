
# Quickstart instructions

## Setting up environment
1) Create conda environment and activate it
    ```console
   conda create --name GCN_MULTI_env python=3.11
   conda activate GCN_MULTI_env
   ```
2) Install requirements
    ```console
   pip install -r requirements.txt 
    ```


## Preprocessing CAMUS DATA
1) Download the CAMUS dataset folder from https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8/folder/63fde55f73e9f004868fb7ac
3) Extract the downloaded folder and place the  ``` database_nifti ``` folder in ``` data/local_data ```
3) Run ```PYTHONPATH=./ python tools/preprocess_CAMUS_displacement.py```

## Evaluation of trained model:
1) Download the trained model from https://huggingface.co/gillesvdv/GCN_with_displacement_camus_cv1
2) place the downloaded .pth file in ``` experiments/logs/CAMUS_displacement_cv_1/GCN_multi_displacement_small/mobilenet2/trained/ ```
2) Run ``` python eval.py ```
3) The results will be saved in the folder
``` experiments/logs/CAMUS_displacement_cv_1/GCN_multi_displacement_small/mobilenet2/trained/weights_CAMUS_displacement_cv_1_GCN_multi_displacement_small_best_loss_eval_on_CAMUS_displacement_cv_1/ ```. The ``` plots ``` subfolder contains the resulting plots and ``` predictions.pkl ``` contains the predictions of 
each sample. 

## Calculating metrics"
See ``` eval/calculate_metrics ``` for an example of how to calculate metrics (Dice and Hausdorff) from the predictions.


## Training your own model:
1) Run ``` python train.py ``` (this will take a long time as the default trains for 5000 epochs)
2) Change the WEIGHTS parameter in ``` files/configs/Eval_CAMUS_displacement.yaml ```
   to the path of checkpoint of the trained model in 
    ``` experiments/logs/your_dataset/mobilenetv2/your_run_id/your_weights.pth ```
    where your_dataset is the name of the dataset you trained on and your_weights is the name of the checkpoint you want to use,
    and your_run_id is the automatically generated id of the run you want to use.








