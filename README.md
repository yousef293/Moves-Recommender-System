# 🎬 Deep Learning-Based Movie Recommender System with NeuMF

This project implements a **hybrid movie recommender system** using **Neural Matrix Factorization (NeuMF)**. It provides personalized movie recommendations to users based on their previous ratings and unseen movies.

The project integrates machine learning with web APIs and database management to simulate a real-world recommendation service.

---

## 📌 Project Objectives

- Build a **Neural Collaborative Filtering** model (NeuMF) to learn user-item interaction patterns.
- Serve model predictions through a **FastAPI** RESTful service.
- Use **MongoDB Atlas** for storing and retrieving movie metadata and user ratings.
- Enable persistent storage by saving the model after training (`.h5` format) so it can be reused without retraining.
- Test and verify endpoints using **Postman**.

---

## 🧠 Model Architecture: NeuMF

NeuMF combines:
- **GMF (Generalized Matrix Factorization)**: captures linear interaction between users and items.
- **MLP (Multi-layer Perceptron)**: captures nonlinear interactions for better generalization.

Final output: predicted rating between a user and a movie.

### Model Inputs
- `userId`: encoded integer for each user
- `movieId`: encoded integer for each movie

### Model Output
- Predicted rating score (float)

---

## 🧪 Workflow

### 1. **Data Preprocessing**
- Datasets: `ratings.csv`, `movies.csv`
- Ratings and movie metadata are imported into **MongoDB Atlas**.
- User and movie IDs are label-encoded.
- Dataset is split into training and validation sets.

### 2. **Model Training Endpoint**
- **Endpoint**: `POST /train_model`
- Trains NeuMF and saves it as `neumf_model.h5` locally.
- Stores ratings and movie data in MongoDB if not already present.

### 3. **Prediction Endpoint**
- **Endpoint**: `GET /predict/{user_id}`
- Loads the saved model and returns top 5 unseen movie recommendations for the given user.
- Predictions are made only on **movies the user hasn't rated yet**.
- Movie titles are retrieved from MongoDB using movie IDs.

### 4. **Postman Testing**
- Both endpoints were tested using **Postman** to ensure the correct functionality of the API.
- JSON responses were validated for structure and accuracy.

---

## 🧰 Tech Stack

| Component     | Technology               |
|---------------|---------------------------|
| Model         | TensorFlow (NeuMF)        |
| API           | FastAPI                   |
| Database      | MongoDB Atlas (cloud)     |
| Data Handling | Pandas, NumPy             |
| Testing       | Postman                   |
| Format        | HDF5 model saving (`.h5`) |

---

## 💾 Persistent Model Usage

After training:
- The model is saved as `neumf_model.h5` locally.
- When the API server restarts, it automatically loads the saved model without needing to retrain.
- This improves performance and enables reuse in production environments.

---

## 🔐 MongoDB Setup

- Cloud MongoDB Atlas is used for persistent storage of movies and ratings.
- Replace your MongoDB URI in `main.py`:
  ```python
  mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<dbname>



