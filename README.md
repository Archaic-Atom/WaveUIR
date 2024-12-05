# WaveUIR: Wavelet-based Guided Transformer Model for Efficient Universal Image Restoration

[Zhidong Zhu](https://scholar.google.com/citations?user=gMoJbSsAAAAJ&hl=zh-CN), Bangshu Xiong, [Zhibo Rao](https://scholar.google.com/citations?user=36YvjLAAAAAJ&hl=zh-CN), Jinhao Zhu, [Wenchao Li](https://scholar.google.com/citations?hl=zh-CN&user=ph9tt1oAAAAJ), Qiaofeng Ou, [Xing Li](https://scholar.google.com/citations?user=8IcXm6AAAAAJ&hl=zh-CN)

The paper is currently under review.

## Model Performance Comparison
<img src = "figs/WaveUIR_Comparison.png"> 

## Data Download and Preparation

All the datasets used in the paper can be downloaded from the following locations:

Denoising: [BSD400](https://drive.google.com/file/d/1idKFDkAHJGAFDn1OyXZxsTbOSBx9GS8N/view?usp=sharing), [WED](https://drive.google.com/file/d/1e62XGdi5c6IbvkZ70LFq0KLRhFvih7US/view?usp=sharing), [Urban100](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u), [Kodak24](https://r0k.us/graphics/kodak/), [BSD68](https://github.com/clausmichele/CBSD68-dataset/tree/master/CBSD68/original)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1-_Tw-LHJF4vh8fpogKgZx1EQ9MhsJI_f?usp=sharing)

Dehazing: Train [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2), Test [SOTS-Outdoor](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)

Deblur: [GoPro](https://drive.google.com/file/d/1y_wQ5G5B65HS_mdIjxKYTcnRys_AGh5v/view?usp=sharing)

Enhance: [LOL-V1](https://daooshee.github.io/BMVC2018website/)

The training data should be placed in ``` data/Train/{task_name}``` directory where ```task_name``` can be Denoise, Derain, Dehaze, Deblur, or Enhance.

The testing data should be placed in the ```test``` directory wherein each task has a seperate directory. 

Details can be found in data_dir/DATA_README.md

## Environment Installation

1. Clone our repository
```
git clone https://github.com/Archaic-Atom/WaveUIR.git
cd WaveUIR
```

2. Create conda environment
The Conda environment used can be recreated using the env.yml file
```
conda env create -f env.yml
```

## Training

After preparing the training data in ```data/``` directory, use 
```
python train.py
```
to start the training of the model. Use the ```de_type``` argument to choose the combination of degradation types to train on. By default it is set to all the 5 degradation tasks (denoising, deraining, dehazing, deblurring, enhancement).

Example Usage: If we only want to train on deraining and dehazing:
```
python train.py --de_type derain dehaze
```

## Testing

After preparing the testing data in ```test/``` directory, place the mode checkpoint file in the ```ckpt``` directory. The pre-trained model can be downloaded [here](https://drive.google.com/drive/folders/19RbxEEEEIKEddudFbIgWYKh3j7PVsdKS). To perform the evaluation, use
```
python test.py --mode {n}
```
```n``` is a number that can be used to set the tasks to be evaluated on, 0 for denoising, 1 for deraining, 2 for dehazing, 3 for deblurring, 4 for enhancement, 5 for three-degradation all-in-one setting and 6 for five-degradation all-in-one setting.

Example Usage: To test on all the degradation types at once, run:

```
python test.py --mode 6
```

## Results
Performance results of the WaveUIR framework trained under the all-in-one setting.

<details>
<summary><strong>Three Distinct Degradations</strong> (click to expand) </summary>

<img src = "figs/WaveUIR_3d_result.png"> 
</details>
<details>
<summary><strong>Five Distinct Degradations</strong> (click to expand) </summary>

<img src = "figs/WaveUIR_5d_result.png"> 
</details><br>

The visual results can be downloaded [here](https://drive.google.com/drive/folders/1CELko-F95W_reox7KzM8HqEip675bABB).

## Citation

## Contact
Should you have any questions, please contact zhidongzhu@buaa.edu.cn

**Acknowledgment:** This code is based on the [PromptIR](https://github.com/va1shn9v/PromptIR) repository.