library(rpart) # library for decision trees
library(caret) # library to split train and test + preprocess
library(e1071) # library to tune the model

# import the data
ds = read.csv("~/M1 MLDM/Data Mining/Project/DataMiningM1/data.csv")

# normalization of the data
scaling = preProcess(ds[,-1], method=c("center", "scale"))
dataset = cbind(ds[,1], predict(scaling, ds[,-1]))
colnames(dataset)[1] = "Bankrupt"

# splite the dataset between train and test
set.seed(3456)
index = createDataPartition(dataset$Bankrupt,p=0.7,list=FALSE,times=1)
train = dataset[index,]
test = dataset[-index,]

# tune the maxdepth parameter
start = Sys.time()
opt_max_depth = tune.rpart(as.factor(Bankrupt) ~ .,
                           data = train, maxdepth = 2:8)$best.parameters$maxdepth
end = Sys.time()
tune_time = end - start
print(tune_time)

# train the model
start = Sys.time()
control <- rpart.control(maxdepth = opt_max_depth)
model <- rpart(as.factor(Bankrupt) ~ ., 
               data = train, 
               control = control)
end = Sys.time()
learning_time = end - start
print(learning_time)

rpart.plot(model)

# evaluate the model
x_test = as.data.frame(test[,-1], drop=false)
y_test = as.factor(test[, 1])
pred_y_test = predict(model, x_test, type="class")

# recall = TP / TP + FN 
# precision = TP / TP + FP
# F1 = 2*Precision*Recall/(Precision + Recall)

conf_matrix = table(pred_y_test, y_test)
print(conf_matrix)
cm = as.data.frame(conf_matrix)

recall_lab_1 = cm$Freq[4] / (cm$Freq[4]+cm$Freq[3])
precision_lab_1 = cm$Freq[4] /(cm$Freq[4]+cm$Freq[2])
f1_lab_1 = 2*precision_lab_1*recall_lab_1/(precision_lab_1 + recall_lab_1)

recall_lab_2 = cm$Freq[1] / (cm$Freq[1]+cm$Freq[2])
precision_lab_2 = cm$Freq[1] /(cm$Freq[1]+cm$Freq[3])
f1_lab_2 = 2*precision_lab_2*recall_lab_2/(precision_lab_2 + recall_lab_2)
accuracy = 100*mean(pred_y_test==y_test)
print(accuracy)
print(f1_lab_1)
print(f1_lab_2)
print(precision_lab_1)
print(recall_lab_1)