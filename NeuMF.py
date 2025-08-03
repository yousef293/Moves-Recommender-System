import tensorflow as tf
from tensorflow.keras.models import Model  #type: ignore
from tensorflow.keras.layers import Input,Dense,Concatenate,Multiply,Flatten,Embedding #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError#type: ignore
from tensorflow.keras.metrics import MeanAbsoluteError#type: ignore





class Recommender_system:
# MF
    def MF(self,ratings,movies):
        ratings=pd.DataFrame(ratings)
        movies=pd.DataFrame(movies)
        print("Ratings columns:", ratings.columns)
        print("Movies columns:", movies.columns)
        print("Sample ratings:", ratings.head())
        print("Sample movies:", movies.head())
        self.dataset=pd.merge(ratings,movies,on='movieId',how='inner')
        self.dataset=self.dataset.drop(columns=['timestamp'])
        self.dataset= self.dataset.applymap(lambda x: x.item() if hasattr(x, 'item') else x)

        self.user_count=self.dataset['userId'].max()
        self.items_count=self.dataset['movieId'].max()
        self.user_input=Input(shape=(1,))
        self.item_input=Input(shape=(1,))
        mf_size=8
        user_embedding=Embedding(self.user_count+1,mf_size)(self.user_input)
        item_embedding=Embedding(self.items_count+1,mf_size)(self.item_input)
        user_vec=Flatten()(user_embedding)
        item_vec=Flatten()(item_embedding)
        mf_vector=Multiply()([user_vec,item_vec])
        return mf_vector,self.dataset
# MLP
    def MLP(self):
        layers=[64,34,16,8]
        mlp_user_embedding=Embedding(self.user_count+1,layers[0]//2)(self.user_input)
        mlp_item_embedding=Embedding(self.items_count+1,layers[0]//2)(self.item_input)
        mlp_user_vec=Flatten()(mlp_user_embedding)
        mlp_item_vec=Flatten()(mlp_item_embedding)
        mlp_vec=Concatenate()([mlp_user_vec,mlp_item_vec])

        for idx,layer_size in enumerate(layers):
            mlp_vec=Dense(layer_size,activation='relu')(mlp_vec)
        return mlp_vec
    def NeuMF(self,ratings,movies):
        mf_vector,data=self.MF(ratings,movies)
        mlp_vec=self.MLP()
        neuMf_vec=Concatenate()([mf_vector,mlp_vec])
        output=Dense(1,activation='linear')(neuMf_vec)
        model=Model(inputs=[self.user_input,self.item_input],outputs=output)
        model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
        model.summary()
        return model,data

    def Train(self,ratings,movies):
        model,data=self.NeuMF(ratings,movies)
        self.dataset=data
        model.fit([self.dataset['userId'], 
                                 self.dataset['movieId']], 
                                 self.dataset['rating'], 
                                 batch_size=128, 
                                 epochs=10, 
                                 validation_split=0.3)
        model.save(r"c:\Users\ygala\OneDrive\Desktop\recommender system/neumf_model.h5")

        return model,data,('succesfully trained')
    def recommended_for_user(self, movie_ids, user_id, model):
        # movie_ids is a list of integers
        user_input_array = np.array([user_id] * len(movie_ids))
        movie_input_array = np.array(movie_ids)

        # Predict ratings
        predictions = model.predict([user_input_array, movie_input_array], verbose=0).flatten()

        # Return a DataFrame with movie IDs and predictions
        return pd.DataFrame({
            'movieId': movie_ids,
            'predicted_rating': predictions
        }).sort_values(by='predicted_rating', ascending=False)




