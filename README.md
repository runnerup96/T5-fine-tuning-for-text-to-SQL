### T5 training with confidence estimation for paper "Confidence Estimation for Error Detection in Text-to-SQL Systems" AAAI-2024

#### Set-up
To set-up environment for traning and inference install Python 3.10 and install requirements.txt with pip. Download relevant T5 model from hugginface.com.


#### Data preparation
To prepare data, run the preparation script with according splits of data(PAUQ, EHRSQL):

```console
python data/prepare_tsv_dataset.py
```


#### Training

To run training, configure paths in run_train_hf.sh file to SFT train and val datasets and hyper-parameters and run with bash:

```console
. run_train_hf.sh
```


#### Inference
To run inference, configure path in infer_hf_t5.sh to SFT test dataset and hyper-paramenters and run with bash:

```console
. infer_hf_t5.sh
```







