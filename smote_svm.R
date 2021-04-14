# install.packages('smotefamily')
# install.packages('caret')
# install.packages('e1071')
library(smotefamily)
library(caret)
library(e1071)


ds = read.csv("~/M1 MLDM/Data Mining/Project/DataMiningM1/data.csv")
scaling = preProcess(ds[,-1], method=c("center", "scale"))
dataset = cbind(ds[,1], predict(scaling, ds[,-1]))
colnames(dataset)[1] = "Bankrupt"

pca = prcomp(dataset[1:96])

#jpeg(file="SMOTE_PCA_fig.jpeg")
#autoplot(pca, data = dataset, colour = 'Bankrupt')
#dev.off()

set.seed(3456)
index = createDataPartition(dataset$Bankrupt,p=0.7,list=FALSE,times=1)
train = dataset[index,]
test = dataset[-index,]


balanced_train_smote = SMOTE(train[,-1], train[,1], K=10) 

# we set the cross validation with 5 folds

tune = tune(svm, as.factor(class)  ~ ., data = balanced_train_smote$data,
            kernel="linear",
            ranges=list(
              # gamma = c(0.01, 0.1, 0.2, 0.5), # use only with radial kernel
              cost = c(0.01, 0.1, 0.2, 0.5, 1, 2)),
            tunecontrol = tune.control(sampling = "cross", cross = 5))
print(tune$best.parameters)

C_optimal = tune$best.parameters$cost
# gamma_optimal = tune$best.parameter$gamma


start = Sys.time()
model = svm(as.factor(class) ~ ., data = balanced_train_smote$data,
            kernel="linear",
            cost=C_optimal,
            #gamma = gamma_optimal, # use only with radial kernel
            class.weights = c("0"=1, "1"=2)) 
end = Sys.time()
learning_time = end - start
print(learning_time)


x_test = as.data.frame(test[,-1], drop=false)
y_test = as.factor(test[, 1])

pred_y_test = predict(model, x_test)

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
