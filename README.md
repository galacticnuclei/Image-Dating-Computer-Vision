# Image Year Prediction

Machine Learning and Pattern Recognition Final Project

# Overview

This project focuses on predicting the year in which an image was captured using a combination of semantic image embeddings and handcrafted numerical image features. Large image archives often contain missing, inaccurate, or loosely estimated timestamps, making organization and verification difficult. Our goal is to build a scalable machine learning pipeline capable of estimating image dates more precisely.

The system combines CLIP image embeddings for high level semantic understanding with handcrafted numerical image statistics. These representations are then used to train multiple traditional machine learning models for temporal prediction. The project is trained on a large scale dataset containing approximately one million images spanning seven decades.

# Motivation

Many historical and institutional archives contain millions of undated photographs. Automatically estimating image years can significantly improve digital archive organization, metadata verification, and historical preservation efforts. Such a system can also help detect inconsistencies in online media and assist individuals in dating old family photographs more accurately.

Potential applications include digital archiving systems, museum collections, historical preservation projects, media verification pipelines, and academic archival platforms.

# Dataset

The dataset contains approximately one million labeled images collected across multiple decades of visual history. Each sample consists of an image URL paired with its ground truth year label.

To maintain balanced temporal distributions, the pipeline performs stratified random sampling across years before training. This helps reduce bias toward heavily represented time periods.

# Methodology

The pipeline begins by loading and cleaning metadata from CSV files. Missing values are removed and year labels are converted into integer targets. Images are then downloaded in parallel using multithreading to accelerate preprocessing, while corrupted or inaccessible files are skipped automatically.

For semantic feature extraction, the project uses the CLIP vision model:

openai/clip vit base patch32

CLIP embeddings capture rich visual semantics such as fashion trends, vehicle designs, architectural styles, scene composition, and photography characteristics. These embeddings provide strong temporal signals that are highly useful for year prediction tasks.

In addition to CLIP embeddings, the pipeline extracts handcrafted numerical image features including pixel statistics, RGB channel distributions, brightness information, and image dimension properties. These low level visual features complement the high level semantic representations learned by CLIP.

The semantic embeddings and numerical features are concatenated into a unified feature matrix. Preprocessing techniques such as standard scaling and PCA dimensionality reduction are then applied before training.

To improve stability and reduce label noise, years are grouped into five year bins. For example, years from 1980 to 1984 belong to one category, while 1985 to 1989 belong to another. This framing transforms the problem into a more robust temporal classification task.

# Models

The project evaluates several traditional machine learning approaches, including:

Random Forest

Linear SVM

SGD Classifier

Logistic Regression

In addition, a smaller neural network baseline implemented in PyTorch is included for comparison.

Training uses stratified five fold cross validation to ensure balanced temporal distributions across folds and to produce more reliable evaluation metrics.

# Evaluation

Performance is evaluated primarily using Mean Absolute Error and classification accuracy within plus or minus five years.

A prediction is considered correct if the predicted year lies within five years of the actual year. This evaluation strategy reflects the practical difficulty of exact year prediction while still maintaining meaningful temporal precision.

The project currently achieves accuracy exceeding 60 percent within the plus or minus five year range, representing a strong improvement over many traditional approaches reported in prior literature.

Among the evaluated models, Logistic Regression trained on combined CLIP embeddings and numerical features performed particularly well. The experiments also demonstrate that traditional machine learning models can outperform a smaller neural network baseline when paired with strong semantic image representations.

# Technologies Used

Python

PyTorch

Scikit Learn

Transformers

Pandas

NumPy

Matplotlib

PIL

OpenAI CLIP

# Repository Structure

Apr23CodeUpdatedUpdated.ipynb

meta.csv

README.md

# Running the Project

First, install the required dependencies:

pip install torch torchvision transformers scikit learn pandas numpy matplotlib pillow tqdm

Next, prepare a CSV file containing image URLs and their associated year labels. Update the dataset paths inside the notebook configuration accordingly.

Finally, launch Jupyter Notebook and execute the notebook cells sequentially.

# Key Design Decisions

CLIP embeddings were chosen because they encode rich semantic visual information learned from large scale internet data. This enables the model to capture temporal cues such as changes in fashion, architecture, vehicle design, and photography trends across decades.

Traditional machine learning models were selected because they train efficiently, require fewer computational resources, and perform surprisingly well when combined with pretrained semantic embeddings.

Five year temporal bins were used because exact year prediction is inherently noisy. Grouping years into short intervals improves classification stability while still preserving useful chronological precision.

# Potential Applications

Historical archive organization

Automated metadata verification

Digital humanities research

Family photo dating services

News and media verification

Museum digitization projects

Large scale archival indexing

# Future Improvements

Possible future extensions include larger transformer based temporal models, multimodal metadata integration, geographic conditioning, self supervised temporal pretraining, and deployment through real time inference APIs or web applications.

License

This project is intended for research and educational purposes.
