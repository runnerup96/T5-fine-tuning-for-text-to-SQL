# T5: Training and Inference Code for PAUQ

This repository contains the code for training and inference of the **T5 model** on the **PAUQ dataset**. It includes setup instructions, data preparation details, and steps for running training and inference.

---

## Table of Contents
1. [Setup](#setup)
2. [Data](#data)
3. [Training](#training)
4. [Inference](#inference)
5. [Contributing](#contributing)
6. [License](#license)
7. [Contact](#contact)

---

## Setup

To set up the environment, ensure you have **Python 3.10** installed. It is recommended to use **Miniconda** for managing the environment.

1. **Install PyTorch**:  
   Install PyTorch with your designated CUDA version. Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for details.

2. **Install Required Libraries**:  
   Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data

The dataset splits for training and evaluation are available from the following sources:

### Datasets
1. **Original PAUQ XSP**  
   - Repository: [PAUQ XSP](https://github.com/ai-spiderweb/pauq)  
   - Contains the database and table information for PAUQ XSP.

2. **Compositional PAUQ Template SSP and PAUQ Test Long SSP**  
   - Google Drive: [Compositional Splits](https://drive.google.com/drive/folders/12cBewVCrBObBb1qgEg1nXHoqq3hHTT7K?usp=sharing)  
   - Code for preparing compositional splits: [Splitting Strategies](https://github.com/runnerup96/splitting-strategies)  

### Preparing Data for T5 Training
To prepare the data for T5 training:
1. Save the data under the folders `pauq` (for standard splits) or `pauq_xsp` (for cross-domain splits).
2. Run the following script to format the dataset for T5 training:
   ```bash
   python data/prepare_tsv_dataset.py --splits_directory pauq --seed 42 --split_name pauq_xsp
   ```

---

## Training

To train the T5 model, follow these steps:

1. Open the `run_train_hf.sh` script.
2. Set up the required paths (detailed in the script comments).
3. Run the script to start training in a TMUX session:
   ```bash
   ./run_train_hf.sh
   ```

### Monitoring Training Progress
To attach to the TMUX session and monitor progress:
```bash
tmux a -t RUN_NAME
```
Replace `RUN_NAME` with the name of your session. To list all active TMUX sessions, use:
```bash
tmux ls
```

---

## Inference

After training, run inference using the following steps:

1. Open the `infer_hf_t5.sh` script.
2. Set up the required paths (detailed in the script comments).
3. Run the script in a TMUX session:
   ```bash
   ./infer_hf_t5.sh
   ```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions, feedback, or collaboration opportunities, feel free to reach out:

- **Telegram**: [@olg_smv](https://t.me/olg_smv)  
- **Email**: [somov.ol.dm@gmail.com](mailto:somov.ol.dm@gmail.com)