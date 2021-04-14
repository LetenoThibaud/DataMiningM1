library(ggfortify) #library to plot with PCA

dataset = read.csv("~/M1 MLDM/Data Mining/Project/DataMiningM1/data.csv")

# sink used to save the summary

#sink("summary.txt")
print(summary(dataset))
#sink()

pca = prcomp (dataset[1:96])
#jpeg(file="PCA_fig.jpeg")
autoplot(pca, data = dataset, colour = 'Bankrupt')
#dev.off()