# AAAI-26 Reproducibility Submission: AAA Adversarial Audio Evaluation

## 1. Environment Setup

We recommend using [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

**Step 1: Create and activate the environment**
```bash
conda create -n mamba_cu118 python=3.10.16
conda activate mamba_cu118
```

**Step 2: Ensure pip version 23.3.2 is installed**
```bash
pip install pip==23.3.2
```

**Step 3: Install torch 2.2.1 with CUDA 11.8 for Python 3.10**
```
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**Step 4: Install all requirements**
```bash
pip install -r requirements.txt
```

---

## 2. Directory Structure

- `evaluate_LA_Original.py` – Evaluate spoof detection on the original ASVspoof2019-LA dataset.
- `evaluate_LA_Perturbed.py` – Evaluate spoof detection on the adversarially perturbed ASVspoof2019-LA dataset.
- `datasets/` – Contains:
    - `ASVspoof2019_LA_train/`, `ASVspoof2019_LA_dev/`, `ASVspoof2019_LA_eval/`
    - `perturbed_LA_train_XLSRMamba/`, `perturbed_LA_dev_XLSRMamba/`, `perturbed_LA_XLSRMamba/`
- `protocols/` – Protocol/trial list files for the LA evaluation.
- `NISQA/` – For perceptual audio quality evaluation (see below).
- `adversarial_training/` – Scripts and model files for adversarial training experiments.
- Supporting `.py` files, model weights, and utility scripts.

---

## 3. Running Evaluation

### Spoof Detection Evaluation
Evaluate on original data:
```bash
python evaluate_LA_Original.py
```
Evaluate on perturbed (adversarial) data:
```bash
python evaluate_LA_Perturbed.py
```
> *Input paths are pre-set in the scripts.*

---

### Perceptual Quality Assessment (NISQA)
1. Change into the NISQA directory:
    ```bash
    cd NISQA
    ```
2. Run NISQA prediction on a folder of audio files:
    ```bash
    python run_predict.py --mode predict_dir --pretrained_model weights/nisqa.tar --data_dir /path/to/folder/with/audios --num_workers 0 --bs 10 --output_dir ./results
    ```
3. Compute average NISQA metrics for the folder:
    ```bash
    python average.py --result_csv ./results/nisqa_predictions.csv
    ```

---

### Evaluation after Adversarial Training
- Use scripts in `adversarial_training/` for evaluating adversarially trained models:
    ```bash
    python adversarial_training/evaluate_LA_Original.py
    python adversarial_training/evaluate_LA_Perturbed.py
    python adversarial_training/FPR_LA_Original.py
    python adversarial_training/FPR_LA_Robust.py
    ```
- Ensure model weights and input paths are correct as set in the scripts.

---

## 4. Datasets

- `ASVspoof2019_LA_train/`, `ASVspoof2019_LA_dev/`, `ASVspoof2019_LA_eval/` — Clean LA audio files.
- `perturbed_LA_train_XLSRMamba/`, `perturbed_LA_dev_XLSRMamba/`, `perturbed_LA_XLSRMamba/` — AAA-perturbed versions for each split.
- Protocol files are in `protocols/`.

---

## 5. Notes

- The code for generating adversarial perturbations is **not included**; only perturbed audios and all evaluation scripts are provided.
- See comments in each script for usage details.
- Change the cuda devices if needed for parallel processing

---
