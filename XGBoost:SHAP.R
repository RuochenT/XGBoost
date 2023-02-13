#library 
rm(list=ls())
library(plyr) 
library(caret) 
library(xgboost)
library(Matrix)
library(ROSE) #oversampling
library(SHAPforxgboost)
library(data.table)
library(dplyr)
library(ggcorrplot)#check correlation
library(mlr)
library(cvms)

#----------- data preprocessing 
heart_2020_cleaned_2 <- read.csv("heart_2020_cleaned 2.csv")
# sample 10,000 observations from the full dataset 
set.seed(1998)
data<- heart_2020_cleaned_2[sample(nrow(heart_2020_cleaned_2), 10000), ] 
prop.table(table(data$HeartDisease)) #unbalaced response variable (same ratio with original data)
sum(is.na(data)) #no missing values
data[sapply(data, is.character)] <- lapply(data[sapply(data, is.character)], as.factor) #turn character to factor
data$Diabetic<-mapvalues(data$Diabetic,from=c("No, borderline diabetes","Yes (during pregnancy)"),
                         to=c("No","Yes"))
str(data) #check again
model.matrix(~0+., data=data) %>% 
  cor(use="pairwise.complete.obs") %>% 
  ggcorrplot(show.diag = F, type="lower", lab=TRUE, lab_size=2) #check correlation

#---------- split training and test set.
set.seed(1998)
trainIndex<- createDataPartition(data$HeartDisease, p = .80, 
                                 list = FALSE, 
                                 times = 1)
train <- data[ trainIndex,]
test  <- data[-trainIndex,]

#------ using oversampling since HeartDisease (response) variable is unbalanced.
train_over <- ovun.sample(HeartDisease~., data = train, method = "over", N = 13000)$data
prop.table(table(train_over$HeartDisease)) #0.56 is "no"

# convert data frame to data table.
setDT(train_over) 
setDT(test)

#-------- one hot encoding for all categorical variables.
labels <- train_over$HeartDisease
ts_label <- test$HeartDisease
new_tr <- model.matrix(~.+0,data = train_over[,-c("HeartDisease"),with=F]) 
new_ts <- model.matrix(~.+0,data = test[,-c("HeartDisease"),with=F])

#-------- convert factor to numeric 
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1

#-------- preparing matrix 
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

#------- default parameters
params <- list(booster = "gbtree", objective = "binary:logistic",
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, 
               subsample=1, colsample_bytree=1)
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, 
                 nfold = 5, showsd = T, stratified = T, print_every_n = 10,
                 early_stop_rounds = 20, maximize = F)
min(xgbcv$evaluation_log$test_logloss_mean) #round100

#------- first default - model training
xgb1 <- xgb.train(params = params, data = dtrain, nrounds = 100, 
                  watchlist = list(val=dtest,train=dtrain), 
                  print_every_n = 10, early_stop_rounds = 20, 
                  maximize = F , eval_metric = "logloss")

#------- model prediction
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

#------- confusion matrix
c<-confusionMatrix (as.factor(xgbpred), as.factor(ts_label)) #acc 85.84 percent

#------- confusion matrix in a nice table 
cfm <- as_tibble(c$table)
cfm_1<-plot_confusion_matrix(cfm, 
                             target_col = "Reference", 
                             prediction_col = "Prediction",
                             counts_col = "n")  #use this for analysis for better interpretation

#------- second model with random search for parameters and update with the best one
best_param <- list()
best_seednumber <- 1998
best_logloss <- Inf
best_logloss_index <- 0

set.seed(1998)
# 10 iterations to search for all combinations (take longer than 25 minutes)
for (iter in 1:10) {
  param <- list(objective = "binary:logistic",  # For binary
                eval_metric = "logloss",      # logloss for binary
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .1),   # Learning rate, default: 0.3
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(5:10, 1), 
                max_delta_step = sample(5:10, 1)    
                
  )
  cv.nround <-  1000
  cv.nfold <-  5 # 5-fold cross-validation
  seed.number  <-  sample.int(10000, 1) # set seed for the cv
  set.seed(seed.number)
  mdcv <- xgb.cv(data = dtrain, params = param,  
                 nfold = cv.nfold, nrounds = cv.nround,
                 verbose = F, early_stopping_rounds = 8, maximize = FALSE)
  
  min_logloss_index  <-  mdcv$best_iteration
  min_logloss <-  mdcv$evaluation_log[min_logloss_index]$test_logloss_mean
  
  if (min_logloss < best_logloss) {
    best_logloss <- min_logloss
    best_logloss_index <- min_logloss_index
    best_seednumber <- seed.number
    best_param <- param
  }
}
xgb3 <- xgb.train(params = best_param, data = dtrain, nrounds = 300, 
                  watchlist = list(val=dtest,train=dtrain), 
                  print_every_n = 10, 
                  maximize = F) 

#------- model prediction with the best parameter
xgbpred3 <- predict (xgb3,new_ts)
xgbpred3<- ifelse (xgbpred3 > 0.5,1,0)
#---------- confusion matrix.
c2<-confusionMatrix (as.factor(xgbpred3), as.factor(ts_label))

#confusion matrix in a nice table 
cfm2 <- as_tibble(c2$table)
cfm_2<-plot_confusion_matrix(cfm2, 
                             target_col = "Reference", 
                             prediction_col = "Prediction",
                             counts_col = "n")  

#---------- SHAP global and local interpretations.
shap_values = shap.values(xgb3, X_train= new_ts)
shap_values$mean_shap_score #calculate from shap_values$shap_score for each row in test_matrix

# To prepare the long-format data(convert dgCMatrix to matrix):
shap_long <- shap.prep(xgb3, X_train = new_ts,top_n=10)
# SHAP summary plot
plot1<-shap.plot.summary(shap_long)
plot1

# dependence plot  
plot2<-shap.plot.dependence(data_long = shap_long, x= "",color_feature = "SexMale")
plot2

# SHAP force plot for local interpretations.
# choose to show top 5 features by setting `top_n = 5`, 
# set 6 clustering groups of observations.  
plot_data <- shap.prep.stack.data(shap_contrib = shap_values$shap_score, top_n = 5, n_groups = 6)
#  zoom in at a location, and set y-axis limit using `y_parent_limit`
which(ts_label == 1) 
which(xgbpred3 == 1) #The model predict observation 201 has heart disease correctly
#  zoom in observation 201 
obs_201<-shap.plot.force_plot(plot_data, zoom_in_location = 201, y_parent_limit = c(-0.1,0.1))

# plot all 6 clusters
groups_6<-shap.plot.force_plot_bygroup(plot_data)
groups_6