#
# Assignment      - Support Vector Machine Assignment
# Author          - Amrita Bhadrannavar
#
# Submission date - 22/04/2018
#
# Filename	      - SVM_Solution_Amrita_DDA1730233.RR
# 
# Assumptions
# Data Set File   - "mnist_test.csv" & "mnist_train.csv are available in the working directory
#
# HW constraints  - System used for model building is of the following configuration:
#                     RAM = 4GB
#                     Processor = Intel Core2 Duo CPU T6600 @ 2.2GHz
#                     Dell Laptop 2009 Model
#                   Due to above HW configurations, only 25% of the train data has been considered
#
#--------------------------------------------------------------------------------------------------#


#
# Load required libraries
#
library(ggplot2)
library(kernlab)
library(doParallel) 
library(caret)
library(dplyr)
library(caTools)
library(gridExtra)


#
# Load & view the Data Set
#

# Load the dataset
mnist_train <- read.csv("mnist_train.csv", stringsAsFactors = FALSE, header = FALSE)
mnist_test  <- read.csv("mnist_test.csv", stringsAsFactors = FALSE, header = FALSE)

# Check the dimensions
dim(mnist_train)          # 60000 obs. of 785 variables
dim(mnist_test)           # 10000 obs. of 785 variables

# View the dataset
View(mnist_train)
View(mnist_test)
# The columns are named as V1, V2.....V785
# Column V1 is the digit value 0-9
# Column V2-V785 is the pixel information of the digit

# Check the structure of dataset
str(mnist_train)
str(mnist_test)


#--------------------------------------------------------------------------------------------------#


#
# Data Cleaning, Manipulation and Exploration
#

# Check for duplicate data
sum(duplicated(mnist_train))  # No duplicates
sum(duplicated(mnist_test))   # No duplicates

# Check for NA values
sum(is.na(mnist_train))       # No NA values
sum(is.na(mnist_test))        # No NA values

# Check for missing values
sum(sapply(mnist_train, function(x) length(which(x == ""))))  # No missing values
sum(sapply(mnist_test, function(x) length(which(x == ""))))   # NO missing values

# Explore the dataset
summary(mnist_train)
summary(mnist_test)

#
# Data check - range of pixel values
# Pixel range is between 0-255. Data has to be verified if values lie in this range
max(mnist_train)        # Max is 255 - In range
min(mnist_train)        # Min is 0 - In range

max(mnist_test)         # Max is 255 - In range
min(mnist_test)         # Min is 0 - In range


#
# Data check - range of digits
# First column gives the digit value. This should be in the range 0-9
colnames(mnist_train)[1] <- "Digit"
max(mnist_train$Digit)  # Max is 9 - In range
min(mnist_train$Digit)  # Min is 0 - In range

colnames(mnist_test)[1]  <- "Digit"
max(mnist_test$Digit)   # Max is 9 - In range
min(mnist_test$Digit)   # Min is 0 - In range

# Convert Digit column to factor
mnist_train$Digit <- as.factor(mnist_train$Digit)
mnist_test$Digit <- as.factor(mnist_test$Digit)


#--------------------------------------------------------------------------------------------------#


#
# EDA
#

# Data available will have to be plotted to see the distribution of obs available for each digit 
# from 0-9. If any digit the obs are less, we would need to ensure proper sampling to be able to 
# create a good model with training data


# Plot train data
ggplot(mnist_train, aes(x=mnist_train$Digit, fill = mnist_train$Digit)) + geom_bar()
# Based on above plot, no of observations is mostly uniformly spread across for each digit

# Digit distribution numbers of train data
summary(mnist_train$Digit)
#    0    1    2    3    4    5    6    7    8    9 
# 5923 6742 5958 6131 5842 5421 5918 6265 5851 5949 



# Plot test data
ggplot(mnist_test, aes(x=mnist_test$Digit, fill = mnist_test$Digit)) + geom_bar()
# Based on above plot, no of observations is mostly uniformly spread across for each digit

# Digit distribution numbers of test data
summary(mnist_test$Digit)
#   0    1    2    3    4    5    6    7    8    9 
# 980 1135 1032 1010  982  892  958 1028  974 1009 


#--------------------------------------------------------------------------------------------------#


#
# Data transformation - Principal Component Analysis (PCA) using Caret Package
#

# Combining train and test data to perform PCA on combined data set
mnist_total <- rbind(mnist_train,mnist_test)
dim(mnist_total)          # 70000 obs. of 785 variables

# Splitting Digit label and pixel data fields. PCA will be performed only on pixel data.
mnist_total_Digit <- mnist_total$Digit
mnist_total       <- mnist_total[,-1]


#
# Zero Variance feature analysis
# Before PCA, we would need remove the Zero Variance features
# Using "nearZeroVar" command, Variance of features & "near zero variance" is checked
# And features which are in "Zerovariance" are removed
zeroVar_list <- nearZeroVar(mnist_total, saveMetrics=TRUE, allowParallel = T)

# Tabulating Zerovariance and non-Zerovariance
table(zeroVar_list$zeroVar)
# FALSE  TRUE 
# 719    65 

# From the table, we can conclude that 65 features(or columns) are Zerovariance features
# and can be removed
mnist_total <- mnist_total[ ,zeroVar_list$zeroVar==FALSE]
dim(mnist_total)          # 70000 obs. of 719 variables - 65 removed


#
# Data Scaling 
# As the pixel values range from 0-255, scaling will have to done inorder to have a range of [0-1]
# which is a normalised scale.
# Pixel data can be divided by 255 to create a normalised scale.
mnist_total <- mnist_total/255


#
# Principal Component Analysis (PCA) using Caret Package
#
# Total no of variables in dataset is 784 which is large no of variables for model building.
# So we would need to reduce the no of variables.
# To reduce the no of variables, PCA - principal component analysis technique will be used.
# Below are some of links refered to understand and implement PCA
#
# Links used for PCA understanding:
# https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/
# https://datascienceplus.com/principal-component-analysis-pca-in-r/
# https://www.r-bloggers.com/principal-component-analysis-in-r/

# Create covariance matrix
mnist_total_cov <- cov(mnist_total)

# Run PCA using prcomp command
mnist_total_pca <- prcomp(mnist_total_cov)


# Plot standard deviations data to find the number of optimum attribtes to be considered
plot(mnist_total_pca$sdev)

# Based on the above plot, optimum value would be within 100
# Ploting between the range 1-100
plot(sort(mnist_total_pca$sdev, decreasing = T)[c(1:100)])

# From the above plot, 60 is a good candidate for optimum number of attributes
# Reducing dataset to 60 attributes
mnist_total_final <- as.matrix(mnist_total) %*% mnist_total_pca$rotation[,1:60]

# Append the digit label column to transformed data
mnist_total_final <- cbind.data.frame(mnist_total_Digit, mnist_total_final)
colnames(mnist_total_final)[1] <- "Digit"


#
# Split the data into train and test data
train_data <- mnist_total_final[1:60000, ]
test_data  <- mnist_total_final[60001:nrow(mnist_total_final), ]
dim(train_data)          # 60000 obs. of 61 variables 
dim(test_data)           # 10000 obs. of 61 variables 


# train_data and test_data will be data sets which will used for model building


#--------------------------------------------------------------------------------------------------#

#
# Data Setup for model building
#

#
# Data sampling
# Due to large size of train data, only 25% of train data will be considered for model building
# Test data which is 10000 obs will be fully used and no sampling will be applied
set.seed(100)
train_25per_sample <- sample.split(train_data$Digit, SplitRatio = 0.25)
train_data         <- train_data[train_25per_sample,]
dim(train_data)           # 15001 obs. of 61 variables


# Train data will be split in 70:30 ratio as training data and validation data
# Validation data can be used for additonal evaluation of our models
train_data_sample <- sample(2, nrow(train_data), replace = TRUE, prob = c(0.7,0.3))
train_data_final  <- train_data[train_data_sample==1, ]
train_validation  <- train_data[train_data_sample==2, ]
dim(train_data_final)     # 10614 obs. of 61 variables
dim(train_validation)     # 4387 obs. of 61 variables


# Plot train and validation data to visualize the distribution of digits
ggplot(train_data_final, aes(x=train_data_final$Digit, fill = train_data_final$Digit)) + geom_bar()
ggplot(train_validation, aes(x=train_validation$Digit, fill = train_validation$Digit)) + geom_bar()


# train_data_final will be training data for model building
# train_validation will be validation data for model building
# test_data will be the testing data for model evaluation


#--------------------------------------------------------------------------------------------------#

#
# SVM Modelling
#


# Register for parallel computing
cl = makeCluster(detectCores())
registerDoParallel(cl)


#---------------------------------------------------------------- 
# Model 01 - Linear model - SVM  at Cost(C) = 1
#---------------------------------------------------------------- 

# Building linear SVM model with C=1
Model_01 <- ksvm(Digit ~ ., data = train_data_final)


# Predicting results on validation data using Model_01
Val_data_pred_01 <- predict(Model_01,train_validation)

# Confusion matrix for validation data
confusionMatrix(Val_data_pred_01,train_validation$Digit)
# Accuracy : 0.959          
# Statistics by Class:
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity           0.97448   0.9758  0.95444  0.92544  0.96386  0.95538  0.97862   0.9626   0.9586   0.9387
# Specificity           0.99697   0.9972  0.99493  0.99466  0.99648  0.99576  0.99647   0.9956   0.9926   0.9937



# Predicting results on test data using Model_01 
Test_data_pred_01 = predict(Model_01,test_data)

# Confusion matrix for test data
confusionMatrix(Test_data_pred_01,test_data$Digit)
# Accuracy : 0.9617          
# Statistics by Class:
#                       Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9888   0.9894   0.9564   0.9495   0.9654   0.9596   0.9718   0.9446   0.9507   0.9386
# Specificity            0.9973   0.9977   0.9949   0.9941   0.9947   0.9954   0.9969   0.9964   0.9947   0.9953


#---------------------------------------------------------------- 
# Model 02 - Linear model - SVM  at Cost(C) = 10
#---------------------------------------------------------------- 

# Building linear SVM model with C=10
Model_02<- ksvm(Digit ~ ., data = train_data_final, C=10)


# Predicting results on validation data using Model_02
Val_data_pred_02 = predict(Model_02,train_validation)

# Confusion matrix for validation data
confusionMatrix(Val_data_pred_02, train_validation$Digit)
# Accuracy : 0.9692          
# Statistics by Class:
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity           0.98376   0.9839  0.96811  0.93640  0.97349  0.96850  0.98575   0.9834  0.95425  0.95343
# Specificity           0.99747   0.9969  0.99569  0.99568  0.99698  0.99750  0.99823   0.9962  0.99465  0.99648



# Predicting results on test data using Model_02
Test_data_pred_02 = predict(Model_02,test_data)

# Confusion matrix for test data
confusionMatrix(Test_data_pred_02,test_data$Digit)
# Accuracy : 0.9689          
# Statistics by Class:
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9908   0.9912   0.9661   0.9584   0.9756   0.9686   0.9739   0.9591   0.9569   0.9465
# Specificity            0.9969   0.9979   0.9957   0.9944   0.9963   0.9956   0.9981   0.9969   0.9967   0.9970


#---------------------------------------------------------------- 
# Model 03 - Non Linear model - SVM with rbfdot kernel
#---------------------------------------------------------------- 

# Building non linear SVM model with rbfbot kernel
Model_03 <- ksvm(Digit~.,data=train_data_final,scale = FALSE,kernel="rbfdot")
Model_03
#  Hyperparameter : sigma =  0.00916511571111898 


# Predicting results on validation data using Model_03
Val_data_pred_03 <- predict(Model_03, train_validation)

# Confusion matrix for validation data
confusionMatrix(Val_data_pred_03, train_validation$Digit)
# Accuracy : 0.959          
# Statistics by Class:
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity           0.97448   0.9758  0.95444  0.92544  0.96386  0.95538  0.97862   0.9626   0.9586   0.9387
# Specificity           0.99697   0.9972  0.99493  0.99466  0.99648  0.99576  0.99647   0.9956   0.9926   0.9937


# Predicting results on test data using Model_03 
Test_data_pred_03 = predict(Model_03,test_data)

# Confusion matrix for test data
confusionMatrix(Test_data_pred_03,test_data$Digit)
# Accuracy : 0.9617          
# Statistics by Class:
#                      Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9888   0.9894   0.9564   0.9495   0.9654   0.9596   0.9718   0.9446   0.9507   0.9386
# Specificity            0.9973   0.9977   0.9949   0.9941   0.9947   0.9954   0.9969   0.9964   0.9947   0.9953


#--------------------------------------------------------------------------------------------------#

#
# Model Evaluation - Cross validation
#


#---------------------------------------------------------------- 
# Hyperparameter tuning and Cross Validation - Linear - SVM
#---------------------------------------------------------------- 

set.seed(100)

# Performing 5 cross validation 
trainControl <- trainControl(method="cv", number=5, verboseIter=TRUE)
metric       <- "Accuracy"
grid         <- expand.grid(C=seq(1,5, by=1))

# Run validation
Sys.time()
fit.svm <- train(Digit~., data=train_data_final, method="svmLinear", metric=metric, tuneGrid=grid, trControl=trainControl)
Sys.time()

# Analyse results
print(fit.svm)
# Support Vector Machines with Linear Kernel 
# 
# 10614 samples
# 60 predictor
# 10 classes: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 8491, 8492, 8490, 8490, 8493 
# Resampling results across tuning parameters:
#   
#   C  Accuracy   Kappa    
# 1  0.9139811  0.9043878
# 2  0.9124735  0.9027163
# 3  0.9088944  0.8987370
# 4  0.9087053  0.8985261
# 5  0.9080460  0.8977931
# 
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was C = 1.

# Plot the results
plot(fit.svm)


# Valdiating the model after cross validation on test data
Eval_Linear_Test <- predict(fit.svm, test_data)
confusionMatrix(Eval_Linear_Test, test_data$Digit)
# Accuracy : 0.9262
# Statistics by Class:
#                       Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9786   0.9815   0.9322   0.9099   0.9430   0.8700   0.9457   0.9261   0.8727   0.8900
# Specificity            0.9943   0.9966   0.9897   0.9865   0.9899   0.9884   0.9947   0.9933   0.9928   0.9918


#---------------------------------------------------------------- 
# Hyperparameter tuning and Cross Validation - Non Linear - SVM
#---------------------------------------------------------------- 

set.seed(100)

# Performing 5 cross validation 
trainControl <- trainControl(method="cv", number=5, verboseIter=TRUE)
metric       <- "Accuracy"
grid         <- expand.grid(.sigma=seq(0.01, 0.03, by=0.01), .C=seq(1, 3, by=1))

# Run validation
Sys.time()
fit.svm_radial <- train(Digit~., data=train_data_final, method="svmRadial", metric=metric, tuneGrid=grid, trControl=trainControl)
Sys.time()

# Analyse results
print(fit.svm_radial)
# Support Vector Machines with Radial Basis Function Kernel 
# 
# 10614 samples
# 60 predictor
# 10 classes: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 8491, 8492, 8490, 8490, 8493 
# Resampling results across tuning parameters:
#   
#   sigma  C  Accuracy   Kappa    
# 0.01   1  0.9560950  0.9512030
# 0.01   2  0.9617484  0.9574856
# 0.01   3  0.9627850  0.9586375
# 0.02   1  0.9632550  0.9591613
# 0.02   2  0.9662704  0.9625123
# 0.02   3  0.9669300  0.9632452
# 0.03   1  0.9614656  0.9571731
# 0.03   2  0.9639152  0.9598957
# 0.03   3  0.9638211  0.9597912
# 
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were sigma = 0.02 and C = 3.

# Plot the results
plot(fit.svm_radial)

# Valdiating the model after cross validation on test data
Eval_NonLinear_Test<- predict(fit.svm_radial, test_data)
confusionMatrix(Eval_NonLinear_Test, test_data$Digit)
# Accuracy : 0.9722
# Statistics by Class:
#                       Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6 Class: 7 Class: 8 Class: 9
# Sensitivity            0.9908   0.9912   0.9719   0.9634   0.9786   0.9765   0.9749   0.9562   0.9651   0.9524
# Specificity            0.9971   0.9986   0.9953   0.9958   0.9971   0.9965   0.9980   0.9974   0.9961   0.9971


# Close the cluster of parallel computing
stopCluster(cl)

#--------------------------------------------------------------------------------------------------#

#
# Summary and Results
#

# S.NO	Model         	                         "Accuracy" "C value" "sigma value"
#  1	  Model_01 - Linear model  with C = 1         0.962	       1	        NA	
#  2	  Model_02 - Linear model  with C = 10 	      0.969	      10	        NA
#  3	  Model_03 - Non-Linear with rbfbot kernel	  0.962	      NA	        NA
#  4	  Cross Validation - Linear SVM	              0.926	     1-5	        NA
#           Best tune  C = 1, Accuracy = 0.913        
#  5	  Cross Validation - Non-Linear SVM           0.972	     1-3	    0.01-0.03
#           Best tune at sigma = 0.02 and C = 3, Accuracy = 0.967	  

# Based on modelling and cross validation performed, optimal model is :
#               Non Linear SVM Model with sigma = 0.02 and C = 3

#--------------------------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------------------------#
