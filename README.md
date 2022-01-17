# BrixIA COVID-19 Project

## What do you find here
Info, code (BS-Net), link to data (BrixIA COVID-19 Dataset annotated with Brixia-score), and additional material related to the [BrixIA COVID-19 Project](https://brixia.github.io/)

## Defs

BrixIA COVID-19 Project: [go to the webpage](https://brixia.github.io/)
Brixia score: a multi-regional score for Chest X-ray (CXR) conveying the degree of lung compromise in COVID-19 patients

BS-Net: an end-to-end multi-network learning architecture for semiquantitative rating of COVID-19 severity on Chest X-rays

BrixIA COVID-19 Dataset: 4703 CXRs of COVID-19 patients (anonymized) in DICOM format with manually annotated Brixia score

## Project paper
Preprint avaible [here](https://arxiv.org/abs/2006.04603)
```
@article{SIGNORONI2021102046,
title = {BS-Net: learning COVID-19 pneumonia severity on a large Chest X-Ray dataset},
journal = {Medical Image Analysis},
pages = {102046},
year = {2021},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2021.102046},
url = {https://www.sciencedirect.com/science/article/pii/S136184152100092X},
author = {Alberto Signoroni and Mattia Savardi and Sergio Benini and Nicola Adami and Riccardo Leonardi and Paolo Gibellini and Filippo Vaccher and Marco Ravanelli and Andrea Borghesi and Roberto Maroldi and Davide Farina},
}2020}
}
```

## Overall Scheme

![Global flowchart](figures/global-flowchart.png "Global flowchart")

Table of Contents
=================

  * [Data](#Datasets)
  * [Getting Started](#getting-started)
  * [License](#license-and-attribution)
  * [Citation](#Citation)
  
## Datasets

### BrixIA COVID-19 Dataset
The access and use, for research purposes only, of the annotated BrixIA COVID-19 CXR Dataset have been granted form the Ethical Committee of Brescia (Italy) NP4121 (last update 08/07/2020).

[**The data can be downloaded from the website https://brixia.github.io/.**](https://brixia.github.io/#get-the-data)


To unpack all the zipped archives, on unix-like system do:
1. Download all the files
2. From the command line call:  `cat *.tar.gz.* | tar -xzv`
3. A folder called dicom_clean will be created with all the unpacked files

Instead, for MS Window:
2. `type *.tar.gz.* | tar xvfz -`

**[Update]** We revised the dataset and removed the DICOM found to have acquisition problems (low quality). The total now is 4695.

### Annotation and CXR from Cohen's dataset

We exploit the public repository by [Cohen et al.](https://github.com/ieee8023/covid-chestxray-dataset) which contains CXR images (We downloaded a copy on May 11th, 2020).

In order to contribute to such public dataset, two expert radiologists, a board-certified staff member and a trainee with 22 and 2 years of experience respectively, produced the related Brixia-score annotations for CXR in this collection, exploiting [labelbox](https://labelbox.com), an online solution for labelling. After discarding problematic cases (e.g., images with a significant portion missing, too small resolution, the impossibility of scoring for external reasons, etc.), the final dataset is composed of 192 CXR, completely annotated according to the Brixia-score system.


*Below a list of each field in the [annotation csv](data/public-annotations.csv), with explanations where relevant*
<details>
 <summary>Scheme</summary>

| Attribute | Description |
|------|-----|
| filename | filename from Cohen dataset |
| from S-A to S-F | The 6 regions annotatated by a Senior radiologist (+20yr expertise), from 0 to 3 
| S-Global | Global score by the Senior radiologist (sum of S-A : S-F),  from 0 to 18
| from J-A to J-F | The 6 regions annotatated by a Junior radiologist (+2yr expertise),  from 0 to 3 
| J-Global | Global score by the Junior radiologist (sum of S-A : S-F),  from 0 to 18
</details>


### Segmentation Dataset
We provide the script to prepare the dataset as described in the Project paper. 

We exploit different segmentation datasets in order to pre-train the extended-Unet module of the proposed architecture. We used the original training/test set splitting when present (as the case of the JSRT database), otherwise we took the first 50 images as test set, and the remaining as training set (see Table below).

<details>
 <summary>Table</summary>

|  | Training-set |  Test-set | Split | 
|------|-----|-----|-----|
|[Montgomery County](https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/) | 88           | 50       | first 50 |      
| [Shenzhen Hospital](https://arxiv.org/abs/1803.01199) | 516          | 50       | first 50 |
| [JSRT database](http://db.jsrt.or.jp/eng.php)     | 124          | 123      | original |
|------|-----|-----|-----|
|Total             | 728           |223      ||
</details>

The data can be downloaded from their respective sites.


### Alignment synthetic dataset
To avoid the inclusion of anatomical parts not belonging to the lungs in the AI pipeline, which would increase the task complexity or introduce unwanted biases, we integrated into the pipeline an alignment block. This exploits a synthetic dataset (used for on-line augmentation) composed of artificially transformed images from the segmentation dataset (see Table below), including random rotations, shifts, and zooms, which is used in the pre-training phase.

The parameters refer to the implementation in Albumentation. In the last column is expressed the probability of the specific transformation being applied.

<details>
 <summary>Additional details</summary>

|    | Parameters (up to) | Probability |
|----|-----|-----|
|Rotation | 25 degree  |    0.8  |
|Scale    | 10%          | 0.8   |
|Shift     | 10%           | 0.8 |
|Elastic transformation  | alpha=60, sigma=12  |   0.2   |
|Grid distortion     | steps=5, limit=0.3 |    0.2  |
|Optical distortion     | distort=0.2, shift=0.05    |     0.2   |
</details>


## Getting Started

### Install Dependencies

The provided code is written for Python 3.x. To install the needed requirements run:
```
pip install -r requirements.txt
```
For the sake of performance, we suggest to install `tensorflow-gpu` in place of the standard CPU version.

Include the `src` folder in your python library path or launch python from that folder.

### Load Cohen dataset with BrixiaScore annotations
```python
from datasets import brixiascore_cohen  as bsc

# Check the docsting for additional info
X_train, X_test, y_train, y_test = bsc.get_data()
```
### Prepare and load the segmentation dataset

To prepare the segmentation dataset either `Montgomery County`, `Shenzhen Hospital`, and `JSRT` datasets must be
downloaded from their websites and unpacked in a folder (for instance `data/sources/`). Than execute:
```bash
 python3 -m datasets.lung_segmentation  --input_folder data/sources/ --target_size 512
```
or just import it (the first time it is executed, it will create the segmentation dataset):

```python
from datasets import lung_segmentation  as ls

# Check the docsting for additional info. The train-set is provided as a generator, while the validation set is
# preloaded in memory.
# `get_data` accepts a configuration dictionary where you can specify every parameter. See `ls.default_config`
train_gen, (val_imgs, val_masks) = ls.get_data()
```

### Prepare and load the alignment dataset
To prepare the alignment dataset, the segmentation one mush be already built (see previous point)

```python
from datasets import synthetic_alignment  as sa

# Check the docsting for additional info. The train-set and validation-set are provided as generators
# `get_data` accepts a configuration dictionary where you can specify every parameter. See `ls.default_config`
train_gen, val_gen = sa.get_data()
```

### Model weights

The model weight and a demo notebook can be found [here](https://drive.google.com/drive/folders/18PF0xpYd4q_M8CJn7TiO4QXCny1PRgJZ?usp=sharing)

### Other steps

Instructions for preparing and loading the Brixia Covid-19 Dataset and the BS-Net will follow (see specific sections for more info). 


## License and Attribution

**Disclaimer**

The BS-Net model and source code, the BrixIA COVID-19 Dataset, and the Brixia score annotations, are provided "as-is" without any guarantee of correct functionality or guarantee of quality. No formal support for this software will be given to users. It is possible to report issues on GitHub though. This repository and any other part of the BrixIA COVID-19 Project should not be used for medical purposes. In particular this software should not be used to make, support, gain evidence on and aid medical decisions, interventions or diagnoses. Specific terms of use are indicated for each part of the project.

###  Data

   - BrixIA COVID-19 dataset: access conditions and term of use are reported on the [dataset website](https://brixia.github.io/#get-the-data).
   - Pulic Cohen dataset: Each image has license specified in the original file by [Cohen's repository](https://github.com/ieee8023/covid-chestxray-dataset) file. Including Apache 2.0, CC BY-NC-SA 4.0, CC BY 4.0. There are additional 7 images from Brescia under a CC BY-NC-SA 4.0 license.
   - Brixia-score annotations for the pulic Cohen's dataset are released under a CC BY-NC-SA 4.0 license.
  
### Code

  - Released under Open Source license.

## Contacts

Alberto Signoroni alberto.signoroni@unibs.it 

Mattia Savardi m.savardi001@unibs.it

## Citations

For any use or reference to this project please cite the following paper.

[NEWS]: this work got accepted at Medical Image Analysis. Available [here](https://doi.org/10.1016/j.media.2021.102046)
```
@article{SIGNORONI2021102046,
title = {BS-Net: learning COVID-19 pneumonia severity on a large Chest X-Ray dataset},
journal = {Medical Image Analysis},
pages = {102046},
year = {2021},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2021.102046},
url = {https://www.sciencedirect.com/science/article/pii/S136184152100092X},
author = {Alberto Signoroni and Mattia Savardi and Sergio Benini and Nicola Adami and Riccardo Leonardi and Paolo Gibellini and Filippo Vaccher and Marco Ravanelli and Andrea Borghesi and Roberto Maroldi and Davide Farina},
}
```
