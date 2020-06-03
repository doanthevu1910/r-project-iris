rm(list = ls())

install.packages("caret", dependencies = TRUE)

install.packages("ellipse")

install.packages("e1071", dependencies = TRUE)

install.packages("kernlab")

library("caret"); library("ellipse"); library("e1071"); library("kernlab")

setwd(paste(getwd(), "/", "r-project-iris", sep = ""))

getwd()

download.file(url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", destfile = "iris.csv")

dataset <- read.csv("iris.csv")

attach(dataset)

colnames(dataset) <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")

head(dataset)

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)

# select 20% of the data for validation
validation <- dataset[-validation_index, ]

# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

#summary for dataset
dim(dataset)

sapply(dataset, class)

head(dataset)

levels(dataset$Species)

percentage <- prop.table(table(dataset$Species)) * 100

cbind(freq=table(dataset$Species), percentage=percentage)

summary(dataset)

#visualization

#split input and output
x <- dataset[, 1:4]
y <- dataset[, 5]

#boxplot for each attribute on one image
par(mfrow = c(1,4))

for(i in 1:4) {
    boxplot(x[, i], main = names(iris)[i])
}

#barplot for class breakdown
par(mfrow = c(1, 1))

plot(y)

#scatterplot matrix
featurePlot(x = x, y = y, plot = "ellipse")

# box and whisker plots for each attribute
featurePlot(x = x, y = y, plot = "box")

# density plots for each attribute by class value
scales <- list(x = list(relation = "free"), y = list(relation = "free"))

featurePlot(x = x, y = y, plot = "density", scales = scales)

#evaluation

# Run algorithms using 10-fold cross validation
control <- trainControl(method = "cv", number = 10)

metric <- "Accuracy"

# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)

# b) nonlinear algorithms

# CART
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)

# kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)

# c) advanced algorithms

# SVM
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)

# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)

#summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.lda)

#predictions

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)

confusionMatrix(predictions, validation$Species)