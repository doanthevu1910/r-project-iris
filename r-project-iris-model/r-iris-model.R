rm(list = ls())

install.packages("caret", dependencies = TRUE)

install.packages("ellipse")

install.packages("e1071", dependencies = TRUE)

install.packages("kernlab")

library("caret"); library("ellipse"); library("e1071"); library("kernlab")

setwd(paste(getwd(), "/", "r-project-iris-model", sep = ""))

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

#evaluation

# Run algorithms using 10-fold cross validation
control <- trainControl(method = "cv", number = 10)

metric <- "Accuracy"

#linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)

# summarize model
print(fit.lda)

# save the model to disk
saveRDS(fit.lda, "./final_model.rds")

# load the model
super_model <- readRDS("./final_model.rds")
print(super_model)

#predictions

# estimate skill of LDA on the validation dataset
predictions <- predict(super_model, validation)

confusionMatrix(predictions, validation$Species)
