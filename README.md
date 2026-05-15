# Image-Year-Prediction
Machine Learning and Pattern Recognition Final Project
# OVERVIEW

This project focuses on predicting the year in which an image was captured using a combination of semantic image embeddings and handcrafted numerical image features.

Large image archives often contain missing, inaccurate, or loosely estimated timestamps. This project aims to provide a scalable way to estimate image dates more precisely using machine learning.

The pipeline combines CLIP image embeddings for semantic understanding with handcrafted numerical features and traditional machine learning models.

The system is trained on a large scale dataset containing approximately one million images spanning seven decades.

# MOTIVATION

Many historical and institutional archives contain millions of undated photographs. Automatically estimating image years can help improve archive organization, verify metadata consistency, detect misleading media, and assist individuals in dating old family photographs.

Potential applications include digital archiving systems, historical preservation, museum collections, and media verification.

# DATASET

The dataset contains approximately one million labeled images.

Each image includes:

Image URL

Ground truth year label

Images span roughly seventy years of visual history.

The pipeline performs stratified random sampling across years to maintain balance in the dataset.

# PIPELINE

Metadata Processing

The metadata CSV file is loaded and cleaned.

Missing values are removed.

Year labels are converted into integer values.

Parallel Image Downloading

Images are downloaded using multithreading for faster preprocessing.

Corrupted or inaccessible images are skipped automatically.

# CLIP Embedding Extraction

The project uses the CLIP vision model:

openai/clip vit base patch32

CLIP embeddings capture high level semantic information such as:

Fashion trends

Vehicles

Architecture

Scene context

Photography styles

# Numerical Feature Extraction

In addition to CLIP embeddings, the project extracts handcrafted image statistics including:

Global pixel mean

Standard deviation

RGB channel statistics

Brightness information

Image dimensions

Color distribution features

Feature Combination

CLIP embeddings and numerical features are concatenated into a single feature matrix.

The pipeline then applies preprocessing techniques such as:

Standard scaling

PCA dimensionality reduction

Temporal Binning

Years are grouped into five year bins to reduce noise and improve classification stability.

Example:

1980 to 1984 belong to one bin

1985 to 1989 belong to another bin

# Model Training

The project compares multiple machine learning models including:

Random Forest

Linear SVM

SGD Classifier

Logistic Regression

A smaller neural network baseline implemented in PyTorch is also included for comparison.

Cross Validation

The training pipeline uses stratified five fold cross validation to improve evaluation reliability and reduce variance.

# Evaluation Metrics

The primary evaluation metrics are:

Mean Absolute Error

Classification accuracy within plus or minus five years

A prediction is considered correct if the predicted year lies within five years of the actual year.

# RESULTS

The best performing models achieve:

More than 60 percent accuracy within plus or minus five years

Competitive mean absolute error values across decades

Strong improvements over earlier traditional approaches

Among the evaluated methods, Logistic Regression using combined CLIP and numerical features performed particularly well.

The project also demonstrated that traditional machine learning models can outperform a smaller neural network baseline when paired with strong semantic embeddings.

# TECHNOLOGIES USED

Python

PyTorch

Scikit Learn

Transformers

Pandas

NumPy

Matplotlib

PIL

OpenAI CLIP

REPOSITORY STRUCTURE

Apr23CodeUpdatedUpdated.ipynb

sample_ready.csv

downloaded_images/

embeddings/

# RUNNING THE PROJECT

Install Dependencies

pip install torch torchvision transformers scikit learn pandas numpy matplotlib pillow tqdm

Prepare Dataset

Provide a CSV file containing image URLs and associated year labels.

Update the dataset paths inside the notebook.

Run the Notebook

Launch Jupyter Notebook and execute the notebook cells sequentially.

# KEY DESIGN DECISIONS

Why CLIP?

CLIP embeddings encode rich semantic visual information learned from large scale internet data. This helps the model capture temporal cues such as fashion evolution, vehicle design changes, architectural styles, and photography trends.

Why Traditional Machine Learning?

Traditional classifiers train faster, require fewer computational resources, and work effectively when paired with pretrained semantic embeddings.

Why Five Year Bins?

Exact year prediction is extremely noisy. Five year intervals improve stability while still preserving useful temporal precision.

# POTENTIAL APPLICATIONS

Historical archive organization

Automated metadata verification

Digital humanities research

Family photo dating services

News media verification

Museum digitization projects

Large scale archival indexing

# FUTURE IMPROVEMENTS

Potential future extensions include:

Larger transformer based temporal models

Multimodal metadata integration

Geographic conditioning

Self supervised temporal pretraining

Web application deployment

Real time inference APIs


LICENSE

This project is intended for research and educational purposes.
