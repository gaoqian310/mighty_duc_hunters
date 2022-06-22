# True Vertical Depth Prediction -- Team "Mighty DUC Hunters"

Authors: Anthony Akhigbe, Dapo Awolayo, Lahcene Kahbour, Jeremy Zhao, Adam Elnahas, Christopher Liu, Qian Gao

Date: August 19th, 2020


## Introduction:

True vertical depth is the vertical distance from a point (usually the final depth) in the well to a point at the surface and this is one of two primary depth measurements, which is important in determining bottomhole pressures and other subsurface characterizations. Most engineers and geoscientists are required to pass the expected well TVDs to the drillers, which might changes as the wells are drilledd in the field. Under such circumstances, machine learning techniques can be used to predict TVDs to improve subsurface characterization and save time. The goal of this part of the “SPE’s Datathon Contest” is to develop data-driven models by processing various available data from several Wells, and use the data-driven models to predict TVDs in several other Wells within same formation. A robust data-driven model will result in low prediction errors, which is quantified here in terms of Root Mean Squared Error by comparing the predicted and the original TVDs.

We are provided with two major datasets: WellHeader_Datathon.csv (train dataset) and Submission_Sample.csv (test dataset). We built a generalizable data-driven models using train dataset and deployed the newly developed data-driven models on test dataset to predict TVDs. The predicted values were submitted in the same format as Submission_Sample.csv, and submit together with this notebook for reproducibility.

## Summary:
- We find some highly-correlated input features to reduce prediction errors and may cause overfitting.

- Combining Regression Kriging and Support vector model yield good results.

- Removing wells with TVD less than KB elevation helps to better reduce prediction errors.

- Different machine learning models for different formation improve the performance of our models.
