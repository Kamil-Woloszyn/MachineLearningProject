#####
#*********************************************************
#**Created By: Kamil Woloszyn                            *
#**K Number: K00273028        				                   *
#**Date: 10/05/2025						                           *
#**Purpose: Implementation of Machine Learning Assignment*
#*********************************************************

#Installing Packages (Run these separately from the rest of the code)
install.packages(c("caret","nnet", "ggplot2", "dplyr", "rpart","rpart.plot", "gridExtra", "doParallel"))

set.seed(9001)
#####
#************************
#**Data Quality Analysis*
#************************

#Importing dataset
data <- read.csv("diabetes_prediction_dataset.csv")

#####
#Checking the data structure of the data set
str(data) #Displaying variable types

#Getting a feel for the data by printing out top and bottom values of the dataset.
head(data)
tail(data)

#Checking Missing Values and checking missing values per each column
print(sum(is.na(data)))
print(colSums(is.na(data)))
  
#Now we have confirmed the data does not miss any values

#Checking statistics of each column
summary(data)

#####
#Visualizing Data to gather information on relationships between data and visualize it's range
#Simple Visualization
barplot(data$age)
hist(data$bmi)
hist(data$HbA1c_level)
hist(data$blood_glucose_level)
hist(data$diabetes)
hist(data$heart_disease)
hist(data$hypertension)
plot(data$bmi,data$blood_glucose_level)
plot(data$bmi,data$age)

#####
#Advanced Visualization using ggplot library
library(ggplot2)

#Density of BMI graph
ggplot(data, aes(x=bmi)) + geom_density(fill="green", alpha=0.5) + theme_bw() + labs(title="Density Plot of BMI", x="BMI", y="Density")

#Waffle Chart of Age vs BMI
ggplot(data, aes(x=factor(age), y=factor(bmi), fill=blood_glucose_level)) + geom_tile() + theme_bw() + labs(title="Waffle Chart of Age vs BMI", x="Age", y="BMI")

#####
#Checking Correlation Between Numeric Values
print(cor(data[, sapply(data, is.numeric)]))

#####
#Checking for outliers
outliersBMI <- boxplot.stats(data$bmi)$out
outliersHbA1cLEVEL <- boxplot.stats(data$HbA1c_level)$out
outliersBLOOD_GLUCOSE_LEVEL <- boxplot.stats(data$blood_glucose_Level)$out
paste("BMI: ",outliersBMI ,"HbA1c Levels: ",outliersHbA1cLEVEL,"Blood Glucose: ",outliersBLOOD_GLUCOSE_LEVEL)
dataCopy <- data

#Example of outliers which could be cut out in BMI
#Calculating threshold and interquartile range
outlier_threshold <- quantile(dataCopy$bmi, probs=c(0.25, 0.75))
iqr_value <- IQR(dataCopy$bmi)

#Setting the upper bounds to 5 times the interquartile range and lower bounds to 2 times the interquartile range (IQR)
lower_bound <- outlier_threshold[1] - 2 * iqr_value
upper_bound <- outlier_threshold[2] + 5 * iqr_value

#Classifying Outliers by adding a new feature to the data
dataCopy$outlier <- ifelse(dataCopy$bmi < lower_bound | dataCopy$bmi > upper_bound, "Outlier", "Normal")

#Visualizing Outliers
ggplot(dataCopy, aes(x=age, y=bmi, color=outlier)) + geom_point(size=3) + scale_color_manual(values=c("Normal"="blue", "Outlier"="red")) +
       theme_minimal() + labs(title="Example Outliers for removal", x="Age", y="BMI")

#I have decided to not do anything about the outliers as they are considered close enough to the bell curve to be deemed useable
#I have decided to not to add any other features to the data set as I believe 8 features will be enough for my purposes. 
#####
#**********************
#**Pre-processing Data*
#**********************
#Normalizing Data
library(dplyr)

#####
#Nested ifelse statement for changing the type of feature into an int to standardize it
data$gender <- ifelse(tolower(data$gender) == "male",0,
               ifelse(tolower(data$gender) == "female",1,2))

#Another nester ifelse statement (trying to limit the dependance on libraries)
data$smoking_history <- ifelse(tolower(data$smoking_history) == "never", 0,
                        ifelse(tolower(data$smoking_history) == "not current", 1,
                        ifelse(tolower(data$smoking_history) == "current", 2,
                        ifelse(tolower(data$smoking_history) == "no info", 3,
                        ifelse(tolower(data$smoking_history) == "former", 4,
                        ifelse(tolower(data$smoking_history) == "ever", 5,6))))))

#####
#Normalizing Calculations resulting in all data being between 0 and 1
data$gender <-              ((data$gender - min(data$gender, na.rm = TRUE))/(max(data$gender, na.rm = TRUE) - min(data$gender, na.rm = TRUE)))
data$age <-                 ((data$age - min(data$age, na.rm = TRUE))/(max(data$age, na.rm = TRUE) - min(data$age, na.rm = TRUE)))
data$smoking_history <-     ((data$smoking_history - min(data$smoking_history, na.rm = TRUE))/(max(data$smoking_history, na.rm = TRUE) - min(data$smoking_history, na.rm = TRUE)))
data$bmi <-                 ((data$bmi - min(data$bmi, na.rm = TRUE))/(max(data$bmi, na.rm = TRUE) - min(data$bmi, na.rm = TRUE)))
data$HbA1c_level <-         ((data$HbA1c_level - min(data$HbA1c_level, na.rm = TRUE))/(max(data$HbA1c_level, na.rm = TRUE) - min(data$HbA1c_level, na.rm = TRUE)))
data$blood_glucose_level <- ((data$blood_glucose_level - min(data$blood_glucose_level, na.rm = TRUE))/(max(data$blood_glucose_level, na.rm = TRUE) - min(data$blood_glucose_level, na.rm = TRUE)))


#Removing all rows containing a null value in the data
#clean_data <- data[complete.cases(data), ]
#print(clean_data)

#####
#**************************************
#**Implementing Machine Learning Models*
#**************************************
#Importing the caret library
library(caret)
data$diabetes <- as.factor(data$diabetes)

#Splitting data into train data and testing data for testing the trained model
TrainingSize <- createDataPartition(data$diabetes ,p = 0.8, list = FALSE)
TrainingData <- data[TrainingSize,]
TestingData <-  data[-TrainingSize,]

#####
#Model 1 - Decision Trees
#Importing the caret package again for quick re running of code
library(caret)
library(rpart)
library(doParallel)
registerDoParallel(cores = 4)
#Defining Hyper Parameters & Training Decision Tree model
model1 <- train(diabetes ~ ., data = TrainingData, method = "rpart",tuneGrid = expand.grid(cp = c(0.05, 0.1, 0.2)),
               trControl = trainControl(method = "cv", number = 3), metric = "Accuracy")

#Printing out the final decision tree
library(rpart.plot)
rpart.plot(model1$finalModel, box.palette = "auto", shadow.col = "grey")

#Using Trained Model to Predict testing data
model1_predictions <- predict(model1, newdata = TestingData)

#Visualizations of prediction results and statistic of model performance
model1_confusionMatrix <- confusionMatrix(model1_predictions, TestingData$diabetes)


#####
#Model 2 - Neural Network
#Importing the nnet package for quick re running of code
library(caret)
library(nnet)

#Defining Hyper Parameters & Training decision tree model
model2 <- train(diabetes ~ ., data = TrainingData, method = "nnet", tuneGrid = expand.grid(size = c(5,20),decay = c(0.001,0.1)),
                trControl = trainControl(method = "cv", number = 3), metric = "none", MaxNWts = 100, maxit = 80, linout = FALSE)

#Using Trained Model to predict testing Data
model2_predictions <- predict(model2, newdata = TestingData, type = "raw")

#Visualization of prediction results and statistic of model performance
model2_confusionMatrix <- confusionMatrix(model2_predictions, TestingData$diabetes)


#####
#*******************************
#**Comparison of the two Models*
#*******************************

library(caret)
library(gridExtra)

#Printing out the two different confusion matrices
print(model1_confusionMatrix)
print(model2_confusionMatrix)

#Printing out statistics on both models side by side
grid.table(cbind("Rows" = rownames(model1_confusionMatrix$overall),"Decision Tree Model" = round(model1_confusionMatrix$overall,4),
    "Neural Network Model" = round(model2_confusionMatrix$overall,4)),theme = ttheme_default())
  
