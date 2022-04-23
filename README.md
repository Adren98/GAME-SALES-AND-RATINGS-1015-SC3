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

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSjifalk1omESSaUXBBKVI16qaoPQYPxya-Sd5Gm__po7WPeP8R3aDBZD-hnYZbWYeSdg&usqp=CAU)

For the first model, we will be using Catboost. CatBoost is a relatively new open-source machine learning algorithm, developed in 2017 by a company named Yandex. One of CatBoost’s core edges is its ability to integrate a variety of different data types, such as images, audio, or text features into one framework. Catboost makes it easy to handle categorical data, opposed to the majority of other machine learning algorithms, that cannot handle non-numeric values. From a feature engineering perspective, the transformation from a non-numeric state to numeric values can be a very non-trivial and tedious task, and CatBoost makes this step obsolete.

CatBoost builds upon the theory of decision trees and gradient boosting. The main idea of boosting is to sequentially combine many weak models (a model performing slightly better than random chance) and thus through greedy search create a strong competitive predictive model.

![](data:image/png;base64,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)

For our second model, we will be using Tensorflow to create a deep neural network model, TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

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
* CatBoost
* 

***

# References

* [Kaggle Meta Dataset](https://www.kaggle.com/datasets/gregorut/videogamesales)
* [VGChartz](https://www.vgchartz.com/)