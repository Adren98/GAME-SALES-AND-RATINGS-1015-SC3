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