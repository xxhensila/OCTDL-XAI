## Credits  
This project is an independent extension of [OCTDL](https://github.com/MikhailKulyabin/OCTDL)  by Mikhail Kulyabin et al.  
The original project is licensed under the MIT License, which is retained in this repository.  

We are not affiliated with the original authors, but acknowledge and build upon their excellent work.

# OCTDL-XAI

# What's Added in This Extension

## 1. Patient-wise k-Fold Cross-Validation
→ Ensures robust performance evaluation by preventing data leakage and revealing generalization across unseen patients.

### To train the model using patient-wise k-fold CV:

```shell
python kfold_train.py -c configs/your_config.yaml
```
#### Optional arguments:

-k or --folds: Number of folds (default: 5)
--save_model: Save best model from each fold

## 2. Training with Lightweight CNN Models
→ Integrated and tested models like MobileNetV2 and EfficientNet-B0, which offer strong accuracy–efficiency trade-offs suited for clinical deployment.


## 3. Explainable AI Integration
→ Implemented EigenCAM tool. Optional -> Grad-CAM & KPCA-CAM to visualize learned attention, improving interpretability and clinical trust in predictions.

### For GradCAM and KPCA-CAM
```shell
python utils/xai.py
```

### For EigenCAM only
```shell
python utils/eigencam.py
```
---

# Optical coherence tomography dataset classification

Framework for 2D classification of the OCT dataset.

Optical coherence tomography (OCT) is an emerging technology for performing high-resolution cross-sectional imaging. 
It utilizes the interferometry concept to create a cross-sectional map of the retina. 
OCT images are two-dimensional data sets that represent the optical backscattering in a cross-sectional plane 
through the tissue. 
OCT allows non-secondary visualization of various structures of the eye, including the retina, vitreous body,
and choroid, and to detect pathological changes in them. 
The study of these images is essential for the diagnosis, treatment, and monitoring of various eye diseases.

The database consists of the following categories and images:

- Age-related Macular Degeneration (AMD): 1231
- Diabetic Macular Edema (DME): 147
- Epiretinal Membrane (ERM): 155
- Normal (NO): 332
- Retinal Artery Occlusion (RAO): 22
- Retinal Vein Occlusion (RVO): 101
- Vitreomacular Interface Disease (VID): 76

For more information and details about the dataset see:
https://rdcu.be/dELrE
https://arxiv.org/abs/2312.08255


Requirements:
- pytorch
- torchvision
- torcheval
- timm
- tqdm
- munch
- packaging
- tensorboard
- omegaconf
- opencv-python
- hydra-core
- scikit-learn



## How to use
In this work we use folder-form dataset structure:
```
├── dataset
    ├── train
        ├── class1
            ├── image1.jpg
            ├── ...
        ├── class2
        ├── class3
        ├── ...
    ├── val
    ├── test
```
Unzip the archive to the directory and use `preprocessing.py` for image preprocessing and dataset preparation:
```
python preprocessing.py
```
Optional arguments:
```
--dataset_folder', type=str, default='./OCT_dataset', help='path to dataset folder')
--labels_path', type=str, default='./OCTDL_dataset/labels.csv', help='path to labels.csv'
--output_folder', type=str, default='./dataset', help='path to output folder')
--crop_ratio', type=int, default=1, help='central crop ratio of image')
--image_dim', type=int, default=512, help='final dimensions of image')
--val_ratio', type=float, default=0.15, help='validation size')
--test_ratio', type=float, default=0.25, help='test size')
--padding', type=bool, default=False, help='padding to square')
```

Training:

```shell
python main.py
```

Optional arguments:
```
-c yaml_file      Specify the config file (default: configs/default.yaml)
-p                Print configs before training
```



## Resources

> Huang, Y., Lin, L., Cheng, P., Lyu, J., Tam, R. and Tang, X., 2023. Identifying the key components in ResNet-50 for diabetic retinopathy grading from fundus images: a systematic investigation. Diagnostics, 13(10), p.1664. [[link](https://www.mdpi.com/2075-4418/13/10/1664)]


## Citation
```
@article{kulyabin2024octdl,
  title={OCTDL: Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods},
  author={Kulyabin, Mikhail and Zhdanov, Aleksei and Nikiforova, Anastasia and Stepichev, Andrey 
          and Kuznetsova, Anna and Ronkin, Mikhail and Borisov, Vasilii and Bogachev, Alexander 
          and Korotkich, Sergey and Constable, Paul A and Maier, Andreas},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={365},
  year={2024},
  publisher={Nature Publishing Group UK London},
  doi={https://doi.org/10.1038/s41597-024-03182-7}
}
```
