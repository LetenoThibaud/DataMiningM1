library(xgboost)
library(caret) # library to split train and test + preprocess

ds = read.csv("~/M1 MLDM/Data Mining/Project/DataMiningM1/data.csv")
scaling = preProcess(ds[,-1], method=c("center", "scale"))
dataset = cbind(ds[,1], predict(scaling, ds[,-1]))
colnames(dataset)[1] = "Bankrupt"

set.seed(3456)
index = createDataPartition(dataset$Bankrupt,p=0.7,list=FALSE,times=1)
train = dataset[index,]
test = dataset[-index,]

dtrain = xgb.DMatrix(data = as.matrix(train[,-1]), label= as.matrix(train[,1]))
dtest = xgb.DMatrix(data = as.matrix(test[,-1]), label= as.matrix(test[,1]))

start = Sys.time()
model = xgboost(data = dtrain, 
                max.depth = 3,
                objective = "binary:logistic",
                nround=10,
                scale_pos_weight = 50
                )
end = Sys.time()
learning_time = end - start
print(learning_time)

pred_y_test = predict(model, dtest) # predict probabilities entre [0;1]
predicted_labels = c(as.numeric(pred_y_test > 0.5)) # classification

conf_matrix = table(predicted_labels, y_test)
print(conf_matrix)
cm = as.data.frame(conf_matrix)

recall_lab_1 = cm$Freq[4] / (cm$Freq[4]+cm$Freq[3])
precision_lab_1 = cm$Freq[4] /(cm$Freq[4]+cm$Freq[2])
f1_lab_1 = 2*precision_lab_1*recall_lab_1/(precision_lab_1 + recall_lab_1)

recall_lab_2 = cm$Freq[1] / (cm$Freq[1]+cm$Freq[2])
precision_lab_2 = cm$Freq[1] /(cm$Freq[1]+cm$Freq[3])
f1_lab_2 = 2*precision_lab_2*recall_lab_2/(precision_lab_2 + recall_lab_2)
accuracy = 100*mean(predicted_labels==y_test)
print(accuracy)
print(f1_lab_1)
print(f1_lab_2)
print(precision_lab_1)
print(recall_lab_1)
