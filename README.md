# Deep Image Segmentation for Breast Contour Detection

## About
Implementation of the paper [_"A novel approach to keypoint detection for the aesthetic evaluation of breast cancer surgery outcomes"_](https://link.springer.com/article/10.1007/s12553-020-00423-8) by Tiago Gonçalves, Wilson Silva, Maria J. Cardoso and Jaime S. Cardoso.

## Clone this repository
To clone this repository, open a Terminal window and type:
```bash
$ git clone https://github.com/TiagoFilipeSousaGoncalves/Deep-Image-Segmentation-for-Breast-Contour-Detection.git
```
Then go to the repository's main directory:
```bash
$ cd Deep-Image-Segmentation-for-Breast-Contour-Detection
```

## Dependencies
### Install the necessary Python packages
We advise you to create a virtual Python environment first (Python 3.7). To install the necessary Python packages run:
```bash
$ pip install -r requirements.txt
```

## Data
To get access to the dataset used in this paper, please send an e-mail to [**tiago.f.goncalves@inesctec.pt**](mailto:tiago.f.goncalves@inesctec.pt).

## Usage
### Generate Train and Test Indices for 5-Fold Cross-Validation
The original [train_test_indices.pickle](data/train-test-indices/train_test_indices.pickle) file is already provide. However, you may generate this file by running:
```bash
$ python generate_train_test_split_indices_cv5.py
```
### ISBI Model
#### Train
First, we need to train the ISBI Model:
```bash
$ python isbi_model_train.py
```
#### Predict
Then, we generate the ISBI Model Predictions (which are needed for the rest of the models):
```bash
$ python isbi_model_predict.py
```

### Hybrid Model
#### Train & Predict
We are then ready to move to the Hybrid Model, which integrates train and prediction in the same script:
```bash
$ python hybrid_model_predict.py
```
We must convert the Hybrid Model predictions to the same notation as ISBI Model predictions (for scoring purposes):
```bash
$ python hybrid_model_reshape_predictions.py
```

### Segmentation Based Model
#### Train U-Net++
This model is based on U-Net++ Model. We first train a U-Net++ Model with our data:
```bash
$ python segmentation_based_model_unetpp_train.py
```
#### Generate Breast Masks with U-Net++
Then, we generate breast masks with the U-Net++ trained model:
```bash
$ python segmentation_based_model_unetpp_predict.py
```
#### Project ISBI Model predictions in the U-Net++ masks detected contours
Finally, we perform contour detection in the U-Net++ predicted masks and combine with the ISBI Model predictions to get a refined breast contour detection:
```bash
$ python segmentation_based_model_predict.py
```

## Scoring and Plots
### Python Scripts
To generate scores you must run the scoring scripts:
#### ISBI Model Scoring
```bash
$ python isbi_model_scoring_results.py
```
#### Hybrid Model Scoring
```bash
$ python hybrid_model_scoring_results.py
```
#### Segmentation Based Model Scoring
```bash
$ python segmentation_based_model_scoring_results.py
```
### Python Jupyter Notebook
To generate scores and to plot the predictions, you may run the [plot_predictions_and_get_scores.ipynb](plot_predictions_and_get_scores.ipynb), using Jupyter-Notebook or [Jupyter-Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html). To install [Jupyter-Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html):
```bash
$ pip install jupyterlab
```
And then run:
```bash
$ jupyter-lab
```

## Citation
If you use this repository in your research work, please cite this paper:
```bibtex
@article{gonccalvesnovel,
  title={A novel approach to keypoint detection for the aesthetic evaluation of breast cancer surgery outcomes},
  author={Gon{\c{c}}alves, Tiago and Silva, Wilson and Cardoso, Maria J and Cardoso, Jaime S},
  journal={Health and Technology},
  pages={1--13},
  publisher={Springer}
}
```

## Credits and Acknowledgments
### ISBI and Hybrid Models
This model and associated [**code**](https://github.com/wjsilva19/k_detection) are related to the paper [_"Deep Keypoint Detection for the Aesthetic Evaluation of Breast Cancer Surgery Outcomes"_](https://ieeexplore.ieee.org/abstract/document/8759331) by Wilson Silva, Eduardo Castro, Maria J. Cardoso, Florian Fitzal and Jaime S. Cardoso.
### U-Net++ Model
This model and associated [**code**](https://github.com/MrGiovanni/UNetPlusPlus) are related to the paper [_"UNet++: A Nested U-Net Architecture for Medical Image Segmentation"_](https://arxiv.org/abs/1807.10165) by Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, and Jianming Liang.