Fully reproducible code for AugUnet1D and methods we compared it against.

# ENVIRONMENT SETUP

- Install Anaconda/Miniforge
- Create environment
    ```conda env create -f environment.yml```
- Activate environment
    ```conda activate mice_eeg_env ```

# AUGUNET1D
- augunet1D/training contains code for training
    - all training code uses PyTorch Lightning based code.
    - use [exp1_final_aug_train_unet.py](./augunet1D/training/exp1_final_aug_train_unet.py) to train your own model.
    - replace ```train_path``` and ```test_path``` in the script to point to your directories containing the training and testing data. Data for training using our data available in .MAT format in Zenodo: https://doi.org/10.5281/zenodo.17982390. Follow instructions in the link provided to find the training and testing data.
    - Run ```python exp1_final_aug_train_unet.py``` in the terminal.
- augunet1D/inference contains code for inference.
    - use the [evaluation_all_mice.ipynb](./augunet1D/inference/evaluation_all_mice.ipynb) for doing inference.
    - Model checkpoint available in folder in [Google Drive](https://drive.google.com/drive/folders/1i3NTwTuxoBsRoQodCZI6-4VYoH4uGafP?usp=drive_link).
    - Data available in .MAT format (used for training) in Zenodo: https://doi.org/10.5281/zenodo.17982390

We adapted code from the following repositories for this project:
- PyTorch 1D UNet model definitions from [https://github.com/fepegar/unet/blob/main/unet/unet.py](https://github.com/fepegar/unet/blob/main/unet/unet.py).
- ECG Augmentations from [https://github.com/klean2050/ecg-augmentations](https://github.com/klean2050/ecg-augmentations).
- Loss functions from [https://github.com/BloodAxe/pytorch-toolbelt/tree/develop/pytorch_toolbelt/losses](https://github.com/BloodAxe/pytorch-toolbelt/tree/develop/pytorch_toolbelt/losses).
- Cosine Warmup Scheduler from [https://github.com/santurini/cosine-annealing-linear-warmup/tree/main/cosine-warmup](https://github.com/santurini/cosine-annealing-linear-warmup/tree/main/cosine-warmup).

We would also like to thank this blog post from [Sebastian Raschka](https://sebastianraschka.com/) for guiding us with PyTorch Lightning https://sebastianraschka.com/blog/2023/pytorch-faster.html.

# COMPARISON CODE
We use github repo provided by DETRtime[^1] authors and modify it for our dataset. Please reference this repository for the original code: [DETRtime](https://github.com/lu-wo/DETRtime).

The repo required our training data to be converted from MAT format to .npz, which is available in the [Google Drive](https://drive.google.com/drive/folders/1i3NTwTuxoBsRoQodCZI6-4VYoH4uGafP?usp=drive_link) link.

- comparison_code/BASELINES contains code adapted from [DETRtime/Baselines](https://github.com/lu-wo/DETRtime/tree/main/Baselines) for the baselines mentioned in our paper.
    - Please modify ```config["data_dir"]``` in [config.py](./comparison_code/Baselines/config.py) to point to the directory containing the training files. The training files used will be provided.
    - Inside [hyperparameters.py](./comparison_code/Baselines/hyperparameters.py) modify the ```mice_eeg``` dictionary entry to run specific baseline models.
    - Use ```python main.py``` inside the Baselines directory while the conda environment is activated to run the baseline code.
    - Use [predict_using_baselines.ipynb](./comparison_code/Baselines/predict_using_baselines.ipynb) for doing inference on our dataset.
- To run the DETRtime comparison use [train_mode.sh](./comparison_code/DETRtime/train_model.sh) and modify the specific arguments. We use the hyperparameters provided by Wolf et al. (2022) [^1] in their paper with modifications for handling our data.
    - Use [evaluate_on_test_set.ipynb](./comparison_code/DETRtime/evaluate_on_test_set.ipynb) to do inference on our dataset.

- For traditional ML models use the following Jupyter notebooks:
    - [Traditional_ML_Models.ipynb](./comparison_code/Traditional_ML_Models.ipynb) for training KNN, Logistic Regression, Decision Tree and Random Forest Models.
    - [Traditional_ML_per_mouse_prediction.ipynb](./comparison_code/Traditional_ML_per_mouse_prediction.ipynb) for inference on each test mice.

# STREAMLIT APP

We also provide an interactive webapp to use our trained models.

To run:
- Run ```cd augunet1D/streamlit_app``` in the terminal.
- With the conda environment activated, run ```streamlit run app.py```.


***

## References

[^1]: Wolf, Lukas, et al. "A deep learning approach for the segmentation of electroencephalography data in eye tracking applications." arXiv preprint arXiv:2206.08672 (2022).