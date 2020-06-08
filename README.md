# Brixia-score-COVID-19
End-to-end learning for semiquantitative rating of COVID-19 severity on Chest X-rays. Additional material and updates.

![Global flowchart](figures/global-flowchart.png "Global flowchart")


## Introduction
In this work we designed an end-to-end deep learning architecture for predicting, on Chest X-rays images (CRX), a multi-regional score conveying the degree of lung compromise in COVID-19 patients.
Such semiquantitative scoring system, namely *Brixia-score*, was applied in serial monitoring of such patients, showing significant prognostic value, in one of the hospitals that experienced one of the highest pandemic peaks in Italy.

We exploited a clinical dataset of almost 5,000 CXR annotated images collected in the same hospital. Our BS-Net demonstrated self-attentive behavior and a high degree of accuracy in all processing stages. 

Through inter-rater agreement tests and a gold standard comparison, we were able to show that our solution outperforms single human annotators in rating accuracy and consistency, thus supporting the possibility of using this tool in contexts of computer-assisted monitoring.

## Data
### Annotation and CXR from Cohen's dataset
Description

### Brixia-dataset
Waiting for Ethical Cometee approval.


## Getting Started
TBD

## Disclaimer
This software, the data, and the annotations, are provided "as-it-is" without any guarantee of correct functionality or guarantee of quality. No formal support for this software will be given to users. It is possible to report issues on GitHub though. This repository and any other part of the BrixIA project should not be used for medical purposes. In particular this software should not be used to make, support, gain evidence on and aid medical decisions, interventions or diagnoses. 
Please do not claim diagnostic performance of a model without a clinical study! This is not a kaggle competition dataset. 


## License and Attribution
Code and annotations:
Data: 

## Contact Information

## Citation

@article{sig2020covid,
  title={End-to-end learning for semiquantitative rating of COVID-19 severity on Chest X-rays},
  author={Alberto Signoroni and Mattia Savardi and Sergio Benini and Nicola Adami and Riccardo Leonardi and Paolo Gibellini and Filippo Vaccher and Marco Ravanelli and Andrea Borghesi and Roberto Maroldi and Davide Farina},
  journal={arXiv ????},
  url={},
  year={2020}
}