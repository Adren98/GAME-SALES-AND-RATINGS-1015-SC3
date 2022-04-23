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

![](https://avatars.mds.yandex.net/get-bunker/56833/dba868860690e7fe8b68223bb3b749ed8a36fbce/orig)

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

* [Kaggle Meta Dataset](https://www.kaggle.com/datasets/deepcontractor/top-video-games-19952021-metacritic)
* [VGChartz](https://www.vgchartz.com/)
* [Python VGChartz Scraper](https://github.com/GregorUT/vgchartzScrape)