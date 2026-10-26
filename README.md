# Visualizing Credit Card Fraud Detection using Gaussian Mixture Models (GMM)

An end-to-end statistical learning and data visualization project focused on credit card fraud detection.

This project was developed in the context of an undergraduate degree in Statistics and investigates fraudulent financial transactions through exploratory data analysis, dimensionality reduction, probabilistic clustering and visual analytics.

The study combines Principal Component Analysis (PCA), t-Distributed Stochastic Neighbor Embedding (t-SNE) and Gaussian Mixture Models (GMM) to explore patterns in highly imbalanced financial data and to investigate whether fraudulent transactions occupy specific regions of the feature space.

---

## Project Motivation

Credit card fraud detection is a challenging statistical problem because fraudulent transactions represent only a very small proportion of all financial transactions.

This imbalance makes traditional analysis difficult and creates several challenges, including:

- severe class imbalance;
- rare anomalous observations;
- overlapping transaction profiles;
- complex relationships between variables;
- high dimensionality;
- non-linear structures;
- heterogeneous behavioral patterns.

Because fraudulent transactions may not form clearly separated groups, this project focuses not only on classification, but also on understanding the latent statistical structure of the data.

The main idea is to investigate whether unsupervised and multivariate statistical methods can reveal meaningful structures associated with fraudulent behavior.

---

## Main Objective

The main objective of this project is to investigate the structure of credit card transaction data and evaluate how dimensionality reduction and probabilistic clustering techniques can support the identification and visualization of fraudulent patterns.

---

## Specific Objectives

The project aims to:

- preprocess and organize a large credit card transaction dataset;
- perform exploratory data analysis;
- investigate the distribution and behavior of transaction-related variables;
- identify potential anomalies and unusual transaction profiles;
- perform feature engineering;
- encode categorical variables appropriately;
- prepare the data for multivariate statistical analysis;
- apply Principal Component Analysis;
- apply t-SNE for non-linear visualization;
- estimate Gaussian Mixture Models;
- identify latent transaction groups;
- analyze probabilistic cluster membership;
- investigate the concentration of fraudulent transactions among clusters;
- compare different low-dimensional representations;
- visually explore anomaly distributions;
- evaluate the statistical usefulness and limitations of the selected methods.

---

## Principal Component Analysis — PCA

Principal Component Analysis is a multivariate statistical technique used to reduce dimensionality while preserving as much information as possible from the original dataset.

PCA transforms the original variables into a new set of orthogonal variables known as principal components.

These components are linear combinations of the original variables and are ordered according to the amount of variance they explain.

Within this project, PCA is intended to support:

- dimensionality reduction;
- visualization of high-dimensional data;
- identification of dominant variability patterns;
- reduction of redundancy between correlated variables;
- investigation of transaction structure;
- preparation of lower-dimensional representations for subsequent analyses.

An important aspect of PCA is the analysis of explained variance.

This allows the project to evaluate how much of the original information can be represented using a smaller number of components.

Component loadings may also be examined to understand which original variables contribute most strongly to each principal component.

---

## t-Distributed Stochastic Neighbor Embedding — t-SNE

t-SNE is a non-linear dimensionality reduction method mainly used for data visualization.

While PCA focuses on preserving global linear variance, t-SNE attempts to preserve local neighborhood relationships between observations.

This characteristic makes t-SNE useful for visually exploring complex structures that may not be visible through linear transformations.

Within this project, t-SNE may help visualize:

- local concentrations of transactions;
- possible anomalous regions;
- groups with similar transaction profiles;
- overlap between legitimate and fraudulent observations;
- structures that may not be evident through PCA;
- relationships between GMM clusters and transaction classes.

t-SNE is used primarily as an exploratory visualization technique.

Its output is therefore interpreted cautiously, since distances and cluster shapes in the final representation should not automatically be interpreted as direct statistical evidence.

---

## Gaussian Mixture Models — GMM

Gaussian Mixture Models are probabilistic models used to represent a dataset as a mixture of multiple Gaussian distributions.

Each Gaussian component represents a latent probabilistic group within the data.

Unlike clustering methods based only on deterministic distance rules, GMM estimates the probability that each observation belongs to each cluster.

This probabilistic characteristic is particularly relevant for financial transaction data because transaction groups may overlap.

A transaction may therefore have partial membership probabilities across multiple latent components.

Within this project, GMM is the central modeling technique.

The method is used to investigate:

- latent transaction profiles;
- probabilistic cluster membership;
- overlapping transaction groups;
- covariance structures;
- low-density observations;
- unusual transaction profiles;
- possible concentrations of fraudulent transactions.

The relationship between the estimated mixture components and the known fraud indicator will later be analyzed to determine whether certain clusters contain disproportionately high concentrations of fraudulent transactions.

---

## Relationship Between PCA, t-SNE and GMM

The three methods play complementary roles in the project.

PCA provides a linear statistical representation of the data.

t-SNE provides a non-linear visual representation focused on local neighborhood relationships.

GMM provides a probabilistic clustering model.

Conceptually:

```text
Original high-dimensional data
          |
          v
      Preprocessing
          |
          v
  Feature transformation
          |
          +-------------------+
          |                   |
          v                   v
         PCA                t-SNE
          |                   |
          v                   v
Linear representation   Visual representation
          |
          v
         GMM
          |
          v
Probabilistic clusters
          |
          v
Fraud distribution analysis
