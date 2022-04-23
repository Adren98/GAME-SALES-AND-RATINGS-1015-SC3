# About
This is a Mini-Project for SC1015 (Introduction to Data Science and Artificial Intelligence) that focuses on predicting the total sales of a game, based on their respective user and critic ratings from VGChartz and Meta. VGChartz and Meta are two renowned websites that critique and review games.

***
# Contributors
* Vignesh Ezhil [@viggy2000](https://github.com/viggy2000): Data merging, Data Scraping, Data cleaning 
* Adren Lim [@adren98](https://github.com/adren98): Machine learning (ML)
* Quek Kar Min [@qkm2000](https://github.com/qkm2000): Exploratory data analysis (EDA)

***

# Problem Definition

* How can we make use of [indicators] to predict the Global Sales of games?

***

# Models Used

![](data:image/png;base64,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)

For the first model, we will be using Catboost. CatBoost is a relatively new open-source machine learning algorithm, developed in 2017 by a company named Yandex. One of CatBoost’s core edges is its ability to integrate a variety of different data types, such as images, audio, or text features into one framework. Catboost makes it easy to handle categorical data, opposed to the majority of other machine learning algorithms, that cannot handle non-numeric values. From a feature engineering perspective, the transformation from a non-numeric state to numeric values can be a very non-trivial and tedious task, and CatBoost makes this step obsolete.

CatBoost builds upon the theory of decision trees and gradient boosting. The main idea of boosting is to sequentially combine many weak models (a model performing slightly better than random chance) and thus through greedy search create a strong competitive predictive model.

![](https://cdn-images-1.medium.com/max/984/1*78KMHVVecnTzkLbN7xW0tQ.png)

TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

Tensorflow gives us more flexibility on how our hidden layers are built, it provides many customizations and allows the user to choose, for example, the number of nodes in each hidden layer, the activation functions, loss function and the optimizer that we want to use for our model. In addition, Keras, which is a high-level neural network library that runs on top of TensorFlow, makes it easier to do.


***

# Stages of the project
[1. Data Scraping and Collation](https://github.com/Adren98/1015/tree/main/Submission/1.%20Data%20Scraping%20and%20Collation) 

[2. Data Cleaning, Merging and Analysis](https://github.com/Adren98/1015/tree/main/Submission/2.%20Data%20Cleaning%2C%20Merging%20and%20Analysis)

[3. Machine Learning](https://github.com/Adren98/GAME-SALES-AND-RATINGS-1015-SC3/tree/main/Submission/3.%20Machine%20Learning)

***

# Conclusion

* Contrary to popular belief, Game Ratings do not very accurately determine game sales
* Since there are weak correlations between ratings, platform and genre and sales, this means that they may not be sufficient to determine the final sales of each game
* Additionally, there are other factors affecting the sales as well, such as marketing strategies and player sentiment 
* Hence, these categories and indicators should not be used as the primary indicators

***

# Skills Learnt

* Collaborating via GitHub
* Collaborating via Google Collab
* CatBoost
* Tensorflow

***

# References

* [Kaggle Meta Dataset](https://www.kaggle.com/datasets/gregorut/videogamesales)
* [VGChartz](https://www.vgchartz.com/)