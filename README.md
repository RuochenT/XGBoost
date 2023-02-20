# Factors behind heart disease 

## Introduction
This project uses Adaboost and XGBoost to predict whether the observation has a heart disease and use SHAP to explain the potential factors behind the result from the best model. 

## Data
The data set originally comes from the Centers for Disease Control and Behavioral Risk Factor Surveillance System. The data was gathered from telephone surveys on the health status and lifestyle of residents in the United States in 2020. The original data set has 319,795 observations with 18 variables with no missing values. In this report, 10,000 observations are randomly selected for analysis. 

## Method

[Adaboost](https://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf)

[XGBoost](https://arxiv.org/abs/1603.02754)

[SHAP](https://arxiv.org/abs/1705.07874)

## Visualization 

The confustion matrix from tuned XGBoost.
![confustion matrix 3-1-1](https://user-images.githubusercontent.com/119982930/220091488-6af9f3e3-bfe6-441c-a273-918624c13662.png)

The summary plot for global interpretation.
![summary plot-1-1](https://user-images.githubusercontent.com/119982930/220091181-ee06e497-1c76-47fc-a5bf-7d61252fd26b.png)

The forceplot for local interpretation (for observation 200 - 350).
![forceplot-1-1](https://user-images.githubusercontent.com/119982930/220091499-50a9ada4-7fc6-4650-9a9d-661e748fad25.png)

## Conclusion
The result from the best model shows that the model has accuracy of 82.8 %, sensitivity of 41.8 %, and specificity of 90.9%. From the summary plot, the five most important factors are BMI, gender, the diﬀiculty of walking or climbing up the stairs, being diabetic, and being 80 years and older. 

## Reference 

Centers for Disease Control and Prevention. (2022, July 12). About Heart Disease. Retrieved from https://www.cdc.gov/heartdisease/about.htm

Chen, T. & Guestrin, C. (2016), XGBoost: A Scalable Tree Boosting System., in Balaji Krishnapu- ram; Mohak Shah; Alexander J. Smola; Charu Aggarwal; Dou Shen & Rajeev Rastogi, ed., ‘KDD’ , ACM, , pp. 785-794.

Lundberg, Scott M., and Su-In Lee. “A unified approach to interpreting model predictions.” Ad- vances in Neural Information Processing Systems (2017).





