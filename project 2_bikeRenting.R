library(caret)
library(randomForest)
library(gbm)
library(ggplot2)
library(corrgram)


setwd("E:/edwisor data science")

df = read.csv("day.csv")
set.seed(1)


miss = data.frame(apply(df, 2, function(x){sum(is.na(x))}))

df$season = as.factor(df$season)
df$yr = as.factor(df$yr)
df$mnth = as.factor(df$mnth)
df$weekday = as.factor(df$weekday)
df$weathersit = as.factor(df$weathersit)
df$workingday = as.factor(df$workingday)


############################# outliers##############################
data_num = df[,10:16]

cnames = colnames(data_num)

## detecting if outliers are present or nots

ggplot(aes_string(y = "cnt"), data = df) + geom_boxplot()
ggplot(aes_string(y = "atemp"), data = df) + geom_boxplot(fill='blue', outlier.color = 'red', outlier.size = 3, outlier.shape = 18)
ggplot(aes_string(y = "temp"), data = df) + geom_boxplot(fill='blue', outlier.color = 'red', outlier.size = 3, outlier.shape = 18)
ggplot(aes_string(y = "hum"), data = df) + geom_boxplot(fill='blue', outlier.color = 'red', outlier.size = 3, outlier.shape = 18)
ggplot(aes_string(y = "windspeed"), data= df) + geom_boxplot(fill='blue', outlier.color = 'red', outlier.size = 3, outlier.shape = 18)


boxplot.stats(data_num$hum)$out
boxplot.stats(data_num$windspeed)$out


############checking correlation ########################


corrgram(data_num, order = F, upper.panel = panel.pie,text.panel = panel.txt, main ="Correlation Plot" )



####################### dropping the features##############################



## dropping atemp variable as it is highly correalted with temp
## dropping "casual" and "registered" variables as their addition is cnt. 
## Treating cnt as target variable
## holidayand workingday seem to explain same thing...dropping holiday
## instant and dteday don't seem to be explaining much. so dropping them



df_final = df[ ,!(names(df) %in% c('casual','registered','atemp','instant','dteday','holiday'))]


set.seed(123)

train_index=createDataPartition(df_final$cnt, p=0.8, list = FALSE)
train=df_final[train_index,]
test=df_final[-train_index,]


X_train = train[ , !(names(train) %in% c('cnt'))]
Y_train = train['cnt']
X_test = test[ , !(names(test) %in% c('cnt'))]
Y_test = test['cnt']



#########################linear regression###########################


model = lm(cnt~.,data= train)


summary(model)

predict_LR = predict(model, test[,1:9])


MAPE= function(x,y)
{
  
  
  mean(abs((x-y)/y))
  
}

MSE = function(x,y)

{

  mean((x-y)^2)
    

  }


MAPE(test[,10],predict_LR)
MSE(test[,10],predict_LR)

######################random forest###########################################


model_RF = randomForest(cnt~. , train, ntree = 500)


predict_RF = predict(model_RF, X_test)


actuals_preds_forest = data.frame(cbind(actuals=test$cnt, predicteds=predict_RF))


MAPE(test$cnt,predict_RF)
MSE(test$cnt,predict_RF)










