# 🖼️ Image-Based Search Engine using CNN and KNN

This project is a **Content-Based Image Retrieval (CBIR)** system that enables users to find visually similar products using image inputs instead of keyword searches. It was developed in response to a real-world business problem where customers were struggling to discover competitively priced items on a retail website, often missing cheaper alternatives that the store actually stocked.

## 🔍 Project Overview

### Context

A retail client approached us after analyzing customer feedback, which revealed a consistent problem:

- Customers were unaware of lower-priced alternatives to premium products available on the site.
- This was largely due to difficulty navigating and searching for items visually.

To address this, we proposed a **proof-of-concept solution** leveraging deep learning and computer vision — building on previous work with CNN-based models — to enable **image-based search functionality**.

## 🧠 Methodology

### 1. Feature Extraction with VGG16

- Implemented pre-trained **VGG16** from Keras applications.
- Replaced the final **MaxPooling** layer with a **Global Average Pooling** layer.
- The network outputs a single, dense **feature vector** per image instead of multidimensional arrays.

### 2. Dataset Processing

- Pre-processed a base set of **300 product images**.
- Extracted and stored their feature vectors using the VGG16 model.
- On querying, the input image is pre-processed and passed through the same model to extract its vector.

### 3. Similarity Matching

- **Cosine Similarity** is used to compare the query vector against the base vectors.
- Returns the top **_N_ most similar images** from the dataset based on the smallest cosine distance.

## 📊 Results

- Tested with multiple example images.
- Visualized the top results along with cosine similarity scores.
- Output results were intuitive and validated the effectiveness of using feature vectors for visual similarity.

## 📈 Discussion & Next Steps

### Current Limitations

The current implementation is a working prototype written as a single script. In production, we would:

- Modularize the code.
- Persist the Nearest Neighbors model and feature vector store.
- Enable real-time updates (adding/removing products from the vector store).
- Return file paths for frontend use (instead of plotting with Matplotlib).

### Future Directions

- Expand to **multiple product categories** with separate feature stores.
- Investigate additional **distance metrics** beyond cosine similarity.
- Evaluate results using **customer feedback** or **click-through rates**.
- Test other pre-trained CNN architectures such as **ResNet**, **Inception**, and **DenseNet**.
