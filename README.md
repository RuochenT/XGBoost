# Factors behind heart disease 

## Introduction
This project uses Adaboost and XGBoost to predict whether the observation has a heart disease and use SHAP to explain the potential factors behind the result from the best model. 

## Data
The data set originally comes from the Centers for Disease Control and Behavioral Risk Factor Surveillance System. The data was gathered from telephone surveys on the health status and lifestyle of residents in the United States in 2020. The original data set has 319,795 observations with 18 variables with no missing values. In this report, 10,000 observations are randomly selected for analysis. 

## Method

### [Adaboost](https://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf)

AdaBoost is one of the ensemble machine learning methods that build a strong model by averaging the weight of weak learners (stumps). The algorithm starts by giving equal sample weight to every observation in the data and then building a tree with one split or a stump from each variable in the data. The algorithm chooses the first stump by calculating the Gini index from each stump and picking the lowest one. Consequently, it calculates the amount of say of that stump to see how well it classifies the observations and uses it to create a new sample weight for building the next stump. If the amount of say is negative, the new sample weight for that observation will be higher since it is misclassified. On the other hand, if it is negative, the new sample weight will decrease since the stump can classify that observation correctly. The algorithm will build the second stump based on the new sample weight of each observation, and the process will keep repeating until the error is low enough or the number of iterations is specified. As a result, the final model or the strong learner will be built using the average weight sample of all weak learners.


### [XGBoost](https://arxiv.org/abs/1603.02754)
XGBoost is one of the ensemble machine learning methods that has been proven to give a state-of-the-art result for both classification and regression in many Kaggle competitions, the Netflix prize, and KDDCup (Chen, 2016). There are three main important features that make it successful. First, the scalability of XGBoost makes it run ten times faster than other current methods. It can handle sparse data with parallel computation. Moreover, it uses out-of-core computations that let a user process a large amount of data that cannot fit into a computer's main memory. Second, it is highly accurate with the test data since it uses a gradient-boosting algorithm with a regularized learning objective that can prevent overfitting. The algorithm will create new models that are made from the previous models' errors and then add all the models together to create a prediction. Third, its feasibility lets users customize their objective and loss function, and let users tune the parameters in the model for each task.

### Tree building algorithm

The problem with the tree structure is to find a tree that improves the prediction along the gradient. Therefore, we need to know how to assign prediction scores and find a good structure of the tree with the gradient descent concept.

The prediction process starts by assigning the data point $x_j$ to the leaf by using directing function and then assigning corresponding score $w_j$ to this data point. As a result, the following function is the final objective function when assigning the score $w_j$ leaf by leaf ($j$-th).

$$Obj^{(t)} = \sum_{j=1}^{T}[(\sum_{i\in I_{j}}g_i)w_j +\frac {1}{2}(\sum_{i\in I_j}h_i + \lambda)w_{j}^2] + \gamma T$$

### [SHAP](https://arxiv.org/abs/1705.07874)

Although tree ensemble methods give high accuracy and fast predictions, it is impossible to understand their internal process. The models are not reliable enough for humans to use in practice, such as in the medical field, financial sector, or scientific research, where understanding the variables that drive each prediction is highly important. Many feature attribution methods, such as LIME, Gini, Saabas, and split count, can help users find global or local interpretations. However, these methods are inconsistent (Lundberg et al., 2017). When the feature in the model changes, the attribution in the explanatory model changes in the opposite direction. Therefore, it is unreliable to compare the attribution values from all features, and it is confusing to find which feature really contributed to the predicted value.

On the other hand, SHAP (SHapley Additive exPlanation) values have been proven to be the only current consistent feature attribution method that can interpret both global and local interpretations. SHAP values are based on Shapley values from cooperative game theory by Lloyd Shapley as a solution to find how much contribution each member that has been working together contributes to producing the final value. For example, a group of five members gets eight points for the presentation grade. We will call a group of cooperating members a coalition or C, and the grade as a coalition value or V. We want to know exactly how much each person contributed to getting this presentation grade. However, most of the time some members will contribute more than others in the group. To find a fair solution, we can compute the Shapley value for each member. By calculating the value for the first member, we remove four other members and then compare the grade between having the first member and not having him help as a part of the group. The difference will be a marginal contribution of member 1 to the coalition consisting of members 2,3,4, and 5. Consequently, we enumerate all possible pairs of coalitions with the difference that based on if member 1 is included. The Shapley value for that member will be the mean calculated from all the marginal contributions. In conclusion, the Shapley value is the average amount of contribution that each member contributes to coalition value. SHAP has the same concept, but it focuses on how much each feature contributes to the model's prediction instead.

SHAP uses additive feature attribution methods which is a linear function to explain complex machine learning models:

$$f(x)=g({x'}) = \phi_0 + \sum_{i=1}^M\phi_i  x'_i$$

Where $\phi_0$ is the average output of the model, $\phi_i$ is like the coefficient that explain effect of feature $i$ or an attribution, $x'_i$ is a discrete binary vector (0 or 1) representing if the feature is included or not, and $M$ is the number of simplified local input features.

According to Lundberg et al (2017), There are three desirable properties of additive feature attribution methods: local accuracy, missingness, and consistency. First, local accuracy states that if the input ($x$) and the simplified input ($x’$) are roughly the same, the actual model $f(x)$ and the explanatory model $g(x’)$ should give the similar output. Second, missingness states that if the feature is not included in the model ($x'_i=0$), its attribution($\phi_i$) must be 0. Therefore, only the features that are included in the model can affect the output of the explanation of the model. Third, the consistency states that when the feature contribution in the model changes, the model or $g(x')$ will change in the same direction.

## Visualization 

### The confustion matrix from tuned XGBoost (the best model)

![confustion matrix 3-1-1](https://user-images.githubusercontent.com/119982930/220091488-6af9f3e3-bfe6-441c-a273-918624c13662.png)

### The summary plot for global interpretation

![summary plot-1-1](https://user-images.githubusercontent.com/119982930/220091181-ee06e497-1c76-47fc-a5bf-7d61252fd26b.png)

The 10 most important variables that have the most impact on heart disease are BMI, SexMale, DiffWalkingYes, DiabeticYes, AgeCategory80 or older, AgeCategory75-79, AgeCategory70-74, GenHealthGood, MentalHealth, and RaceWhite. If the observations were male, who have difficulty in walking, have diabetes, are older than 70 yearsold, or are white, they will have an increase in a log odds ratio or more chance to have heart disease on average. Moreover, having a more severe mental health can lead to an increase in a log odds ratio of heart disease, since the color of the feature value is getting darker as the scale of mental health increases. On the other hand, observations that have a very low BMI show the opposite impact on heart disease.


### The forceplot for local interpretation (for observation 200 - 350)

![forceplot-1-1](https://user-images.githubusercontent.com/119982930/220091499-50a9ada4-7fc6-4650-9a9d-661e748fad25.png)

The 201st observation is predicted correctly to have heart disease. The most important feature that leads this observation to have heart disease is diabetes, and there are other variables and the rest impact is from other variables that are not in the top 5 features from the plot.

## Conclusion
The result from the best model shows that the model has accuracy of 82.8 %, sensitivity of 41.8 %, and specificity of 90.9%. From the summary plot, the five most important factors are BMI, gender, the diﬀiculty of walking or climbing up the stairs, being diabetic, and being 80 years and older. 

## References

Centers for Disease Control and Prevention. (2022, July 12). About Heart Disease. Retrieved from https://www.cdc.gov/heartdisease/about.htm

Chen, T. & Guestrin, C. (2016), XGBoost: A Scalable Tree Boosting System., in Balaji Krishnapu- ram; Mohak Shah; Alexander J. Smola; Charu Aggarwal; Dou Shen & Rajeev Rastogi, ed., ‘KDD’ , ACM, , pp. 785-794.

Lundberg, Scott M., and Su-In Lee. “A unified approach to interpreting model predictions.” Ad- vances in Neural Information Processing Systems (2017).

Lundberg, Scott M., Gabriel G. Erion, and Su-In Lee. “Consistent individualized feature attribution for tree ensembles.” arXiv preprint arXiv:1802.03888 (2018)



