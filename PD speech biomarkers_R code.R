#Speech features of Parkinson's disease patients, patients with REM sleep behavior disorder and healthy controls

#Tasks: exploratory data analysis and predictive modelling

#read data
setwd("C:/R_neuro")
df = read.csv("dataset.csv")
dim(df) # dimensions of dataframe (sample size and variables)
names(df) # list feature names
names(df)[42:65] = paste0("SF", seq(1,24)) # rename speech fetures with shorter names

#variables
pred.dem = names(df[,2:3]) # age and gender
pred.hist = names(df[4:6]) # health history
pred.meds = names(df[7:12]) # medications
mot.exam = names(df[,13:14]) # clinical motor examination scores
symptoms = names(df[,15:41]) # UPDRS III motor scale items (symptoms)
pred.voice = names(df[,42:65]) # only 24 speech features (12 features x 2 speech tests)

#add new varialbes (labels)
df$label = substring(df$Participant..code, 1,2) #define outcome/lalbe/target - diagnostic category

# 1. Exploratory analysis
# 1.1. Univariate analysis (Descriptive statistics)
library(psych)
library(Hmisc)
library(ggplot2)
df$a = "i" #create pseudo group
describe(df$Gender)
ggplot(df, aes(x="", y=Gender, fill=Gender)) + geom_bar(width = 1, stat = "identity") + coord_polar("y", start=0) + 
  theme_void() + xlab("") + scale_fill_manual(values=c("salmon", "royalblue")) # plot piechart

describeBy(df$Age...years., group = df$a, mat=T, digits=3)[-c(1:4,8,9,15)]
qplot(Age...years., data = df, geom = "histogram")
qplot(Age...years., data = df, geom = "density", color = Gender) #age-gender density plots

table(is.na(df[,c(pred.voice)])==T)
ds = describeBy(df[,c(pred.voice)], group=df$a, mat=T, digits=3) # discriptive statistics of speech features 
ds[,-c(1:4,8,9,15)] #show results (hide some statistics)

#hist(df[,c(pred.voice[1:12])]) # alternative plots
#hist(df[,c(pred.voice[13:24])]) # alternative plots

hist.list1 = lapply(
 pred.voice[1:12], 
  function(n) 
    ggplot(data = df, aes_string(x = n)) + 
    geom_histogram(fill="royalblue3") + theme_bw()
)
cowplot::plot_grid(plotlist = hist.list1)

hist.list2 = lapply(
  pred.voice[13:24], 
  function(n) 
    ggplot(data = df, aes_string(x = n)) + 
    geom_histogram(fill="royalblue3") + theme_bw()
)
cowplot::plot_grid(plotlist = hist.list2)

#Kolmogorov-Smirnov Tests - normality of distribution
kst.results = NULL
for (i in pred.voice) {
  kst = ks.test(df[,c(i)],"pnorm", mean(df[,c(i)]), sd(df[,c(i)]))$p.value
  kst.results = rbind(kst.results, cbind(i, kst))
}
kst.results = as.data.frame(kst.results)


table(df$Positive..history..of..Parkinson..disease..in..family)
#replace "-" with NAs throughtout the dataset
df[df == "-"] <- NA

ds2 = describe(df[,c(pred.hist, mot.exam, symptoms)])
ds2

table(is.na(df$Age..of..disease..onset...years.)==T, df$label)
table(is.na(df$X28...Posture)==T, df$label)
table(is.na(df$Overview..of..motor..examination...Hoehn.....Yahr..scale.....)==T, df$label)

for (i in 1:6) { 
  print(pred.meds[i])
  print(table(df$label,df[,pred.meds[i]])) }



library(corrplot)

cor2 <- rcorr(as.matrix(df[,pred.voice]), type = "spearman")
df.cor <- cor2$r # coefficients
p.mat <- cor2$P # p-values

corrplot(df.cor, type="lower")

library(RColorBrewer)
col <- colorRampPalette(brewer.pal(9, "RdBu"))
corrplot(df.cor, method="color", col=col(100),  
         type="lower", 
         addCoef.col = "black",
         tl.col="black", tl.srt=45, 
         p.mat = p.mat, sig.level = 0.01, insig = "blank", 
         diag=F, number.cex=0.5, tl.cex = 0.8 
)

corrplot(df.cor, method="color", col=col(100),  
         type="lower", order="hclust", hclust.method = "centroid",
         tl.col="black", tl.srt=45, 
         p.mat = p.mat, sig.level = 0.01, insig = "blank", 
         diag=F, number.cex=0.5, tl.cex = 0.8 )

heatmap(x = df.cor, col = col(20), symm = TRUE)


pairs.panels(df[,c("SF4", "SF5", "SF16", "SF17", "SF19")], 
             method = "spearman", # correlation method
             hist.col = "royalblue3",
             density = F,  # show density plots
             ellipses = F, # show correlation ellipses
             smooth = F, pch=19, cex.cor=0.6,cex=0.8)

pairs.panels(df[,c("SF13", "SF14", "SF18", "SF22")], 
             method = "spearman", # correlation method
             hist.col = "royalblue3",
             density = F,  # show density plots
             ellipses = F, # show correlation ellipses
             smooth = F, pch=19, cex.cor=0.6,cex=0.8)

library(factoextra)
set.seed(123)
pc <- prcomp(df[,pred.voice], scale.=T)
t(summary(pc)$importance) # importance of components (Proportion of explained variance)
fviz_screeplot(pc, ncp=24,barfill = "royalblue3") # display 24 components due to 24 features used in total
# 16 components explain >95% of total features variability
df.scores = pc$x # save individual data with component scores 
#Plot component loadings
corrplot(t(pc$rotation), tl.col="black")


fviz_contrib(pc, choice = "var", axes = 1:16,fill = "royalblue3")

## Predictive modelling
### 2.1. PD patients -vs- healthy controls  
library(caret)
library(MLeval)
library(doParallel)
df2 = df[df$label!="RB",]
df2$PD = ifelse(df$label == "PD", "case", "control")
algs = c("logreg", "nb", "knn","rf", "xgbDART")

set.seed(123)
cvIndex <- createFolds(factor(df2$PD), 5, returnTrain = T)
k.fold <- trainControl(index = cvIndex,method="cv", number = 5,
                              savePredictions = "final", returnResamp = "final",classProbs=T)
set.seed(123)
LOOCV <- trainControl(method="LOOCV", savePredictions = "final",classProbs=T)


cla <- makePSOCKcluster(6) #Parallel computing - set available number of CPU cores
registerDoParallel(cla)
n=0
perf.res = NULL
for (i in algs) {
n=n+1 
if (n%%1==0) cat("\n","model",n,format(Sys.time()),"\n")

set.seed(987)
mod1 <- train(PD~., method=as.name(i), data = df2[,c("PD", pred.voice)],
              trControl=k.fold ,na.action  = na.pass) 
auroc1 <- evalm(mod1, positive = "case", plots = "r")$stdres[[1]]$Score[13]
CM1 = c(i, "k-fold", confusionMatrix(mod1$pred$pred, mod1$pred$obs)$overall[c(1:2,5)], confusionMatrix(mod1$pred$pred, mod1$pred$obs)$byClass[c(1:4,7)], AUC = auroc1)

n=n+1 
if (n%%1==0) cat("\n","model",n,format(Sys.time()),"\n")

set.seed(987)
mod2 <- train(PD~., method=as.name(i), data = df2[,c("PD", pred.voice)],
              trControl=LOOCV ,na.action  = na.pass) 
auroc2 <- evalm(mod2, positive = "case", plots = "r")$stdres[[1]]$Score[13]
CM2 = c(i, "LOOCV", confusionMatrix(mod2$pred$pred, mod2$pred$obs)$overall[c(1:2,5)], confusionMatrix(mod2$pred$pred, mod2$pred$obs)$byClass[c(1:4,7)], AUC = auroc2)

res = rbind(CM1, CM2)
perf.res = rbind(perf.res, res)
  }
stopCluster(cla)


#XGBoost performs better nthan other algorithms, will use only this method for other tasks

### 2.2. PD and RBD patients -vs- healthy controls 
df.c$neurodeg = ifelse(df.c$label == "HC", "control", "case") # create new class variable

set.seed(123)
cvIndex <- createFolds(factor(df.c$neurodeg), 10, returnTrain = T)
k.fold <- trainControl(index = cvIndex,method="cv", number = 10,
                       savePredictions = "final",classProbs=T) # setup cross-validation method: k-fold

cla <- makePSOCKcluster(6) #Parallel computing - set available number of CPU cores
registerDoParallel(cla)
set.seed(987)
mod3 <- train(neurodeg~., method="xgbDART", data = df.c[,c("neurodeg", "SF.cluster.4", pred.voice)],
              trControl=k.fold ,na.action  = na.pass) 
stopCluster(cla)

auroc3 = evalm(mod3, positive = "case", plots = "r")
confusionMatrix(mod3$pred$pred, mod3$pred$obs)





test = df[df$label == "RB",]
symptoms2 = names(test[,c(15:21,24:41)])
test[,symptoms2] <- sapply(test[,symptoms2], as.character)
test[,symptoms2] <- sapply(test[,symptoms2], as.numeric) 
test$UPDRS.v2 = rowSums(test[,symptoms2])
test$label = ifelse(test$UPDRS.v2 > 3, "symptomatic", "asymptomatic")
table(test$label)

# predict in RBD patients
cvIndex <- createFolds(factor(df2$label), 10, returnTrain = T)
k.fold <- trainControl(index = cvIndex, 
                       method="cv", number = 10,
                       savePredictions = "final", classProbs=T) # setup

#Feature selection
library(plyr)
#T-test for feature selection
ttest <- ldply(pred.voice,  function(colname) {
  t_val = t.test(df2[[colname]] ~ df2[,c("label")])$p.value
  return(data.frame(colname=colname, p.value=t_val))})
preds.add1 <- subset(ttest, p.value<0.1)
#Kruscal test for feature selection
preds.add2 = NULL 
for (m in pred.voice) {
  ktest = kruskal.test(df2[,c(m)], df2[,c("label")])
  if (ktest$p.value < 0.1) 
    preds.add2 = c(preds.add, m)
}
#vector with selected features
preds.select <- unique(c(preds.add2, as.character(preds.add1$colname)))
preds.select2 = c("SF2", "SF4", "SF7", "SF16", "SF17", "SF19", "SF22")
preds.select3 = c("SF2", "SF17", "SF4", "SF16", "SF7", "SF19", "SF22")

set.seed(123)
cvIndex <- createFolds(factor(df2$label), 5, returnTrain = T)
k.fold <- trainControl(index = cvIndex,method="repeatedcv", 
                       number = 5, 
                       savePredictions = "final", returnResamp = "final", 
                       classProbs=T, summaryFunction = twoClassSummary) # setup cross-validation method: k-fold

cla <- makePSOCKcluster(6) #Parallel computing - set available number of CPU cores
registerDoParallel(cla)
set.seed(987)
mod4 <- train(label~., method="xgbDART", data = df2[,c("label", pred.voice)],
              trControl=k.fold, na.action = na.pass, metric = "Spec", maximize = T)
stopCluster(cla)
confusionMatrix(mod4$pred$pred, mod4$pred$obs, positive = "PD")
preds <- predict(mod4, newdata=test)
confusionMatrix(preds, as.factor(test$label), positive = "PD")


# multiclass
set.seed(123)
cvIndex <- createFolds(factor(df.c$label), 5, returnTrain = T)
k.fold <- trainControl(index = cvIndex,method="cv", number = 5, 
                       savePredictions = "final",  
                       classProbs=T) # setup cross-validation method: k-fold

df.c$SF.cluster.4 = as.factor(df.c$SF.cluster.4)
cla <- makePSOCKcluster(6) #Parallel computing - set available number of CPU cores
registerDoParallel(cla)
set.seed(987)
mod4 <- train(label~., method="xgbDART", data = df.c[,c("label","SF.cluster.4", pred.voice)],
              trControl=k.fold, na.action = na.pass)
stopCluster(cla)

confusionMatrix(mod4$pred$pred, mod4$pred$obs)

## 3. Predictive modelling: neurodegeneration severity prediction (numeric prediction)
df3 = df[df$label != "HC",] # dataset with only PD and RBD patients 

df3$UPDRS.score = as.numeric( as.character( df3$Overview..of..motor..examination...UPDRS..III..total.....)) #rename outcome
set.seed(123)
k.fold <- trainControl(method="cv", 
                       number = 10, 
                       savePredictions = "final", returnResamp = "final") # setup cross-validation method: k-fold

cla <- makePSOCKcluster(6) #Parallel computing - set available number of CPU cores
registerDoParallel(cla)
set.seed(987)
mod5 <- train(UPDRS.score~., method="xgbDART", data = df3[,c("UPDRS.score", pred.voice)],
              trControl=k.fold, na.action = na.pass)
stopCluster(cla)

mod5$resample
cor.test(mod5$pred$pred, mod5$pred$obs)
qplot(mod5$pred$pred, mod5$pred$obs)+ geom_smooth()


#k-mean clustering of symptoms
# 1. recoding of symptoms (binary - yes/no)
df.c2 = df3
df.c2[,symptoms] <- sapply(df.c2[,symptoms], as.character)
df.c2[,symptoms] <- sapply(df.c2[,symptoms], as.numeric) 
dichotomize <- function(x, ...){ifelse(x>0, 1, 0)} # custom function to dichotomize values, due to overall rare data and low actual variability 
df.c2[,symptoms] = sapply(df.c2[,symptoms], dichotomize) # dichotomize
symptom.preval = colSums(as.matrix(df.c2[,symptoms]))/80 # compute prevalence of symptoms
symptoms3 = names(which(symptom.preval>0.4)) #selecting only prevalent symptoms (appeared in at least 40% of participants). Rare are symptoms highly likely appear only in more severy PD patients and adding them into k-means clustering will resilt in clustering by the overall severity score rather than by potential subtypes based on symptom patterns (co-occuring symptoms)

# Find optimal number of clusters: Elbow method
fviz_nbclust(as.matrix(df.c2[,symptoms3]), kmeans, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method")

set.seed(123)
clan <- kmeans(as.matrix(df.c2[,symptoms3]), 4, nstart = 25) # k-mean clustering (k=4)
df.c2$cluster.4 = as.factor(clan$cluster)
levels(df.c2$cluster.4) = c("C1", "C2", "C3", "C4")

#Multi Dimensional Scaling
d <- dist(df.c2[,symptoms3]) # euclidean distances between the rows
mds <- cmdscale(d,eig=TRUE, k=2) # k is the number of dim

# plot solution
x <- mds$points[,1]
y <- mds$points[,2]
plot(x, y, xlab="Dim 1", ylab="Dim 2",
     main="Multi Dimensional Scaling", type="n")
text(x, y, labels = df.c2$label, cex=0.8, col = df.c2$col.4)
df.c2$col.4 = ifelse(df.c2$cluster.4=="C1", "gray10", ifelse(df.c2$cluster.4=="C2", "green", ifelse(df.c2$cluster.4=="C3", "red", "blue")))
ifelse(df.c2$cluster.4=="C2", "green", ifelse(df.c2$cluster.4=="C3", "red", "blue"))
# multiclass
set.seed(123)
cvIndex <- createFolds(factor(df.c2$cluster.4), 4, returnTrain = T)
k.fold <- trainControl(index = cvIndex,method="cv", 
                       number = 4,
                       savePredictions = "final", 
                       classProbs=T) # setup cross-validation method: k-fold

cla <- makePSOCKcluster(6) #Parallel computing
registerDoParallel(cla)
set.seed(987)
modA <- train(cluster.4~., method="xgbDART", data = df.c2[,c("cluster.4", pred.voice)],
              trControl=k.fold, na.action = na.pass)
stopCluster(cla)

confusionMatrix(modA$pred$pred, modA$pred$obs)





#k-mean clustering of speech features
# 1. normilize within 0-1 range of minimax
df.c = df
range01 <- function(x, ...){(x - min(x, ...)) / (max(x, ...) - min(x, ...))} # custom rescale function
df.c[,pred.voice] = sapply(df.c[,pred.voice],range01) # rescale values between 0 and 1

# Find optimal number of clusters: Elbow method
fviz_nbclust(as.matrix(df.c[,pred.voice]), kmeans, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method")

set.seed(123)
clan <- kmeans(as.matrix(df.c[,pred.voice]), 4, nstart = 25) # k-mean clustering (k=4)
df.c$SF.cluster.4 = clan$cluster

#MDS
d <- dist(df.c[,pred.voice], method = "euclidean") # euclidean distances between the rows
mds.sf <- cmdscale(d,eig=TRUE, k=2) # k is the number of dim

# plot solution
df.c$x <- mds.sf$points[,1]
df.c$y <- mds.sf$points[,2]

ggplot(data = df.c, aes(x = x, y = y, color=label)) + 
  geom_point(size=2) +
  theme_bw() + ylab("Dim 2") + xlab("Dim 1") + ggtitle("Multidimensional Scaling") +
  scale_color_manual(values=c("gray50", "tomato", "gold"))

ggplot(data = df.c, aes(x = x, y = y, shape=label, color=as.factor(SF.cluster.4))) + 
  geom_point(size=2) +
  theme_bw() + ylab("Dim 2") + xlab("Dim 1") + ggtitle("Multidimensional Scaling") +
  scale_shape_manual(values=c(19,2,0)) +
  scale_color_manual(values=c("green2", "magenta2", "mediumblue", "red"))



