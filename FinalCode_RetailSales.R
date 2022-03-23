# Submitted by: 

# Sayali Mahamulkar

# Importing libraries.
library(forecast)
library(ggplot2)
library(zoo)

#setwd("~/Desktop/Sayali/MSBA/SEM 4 (Spring 2022)/BAN673TSA/TSDProject")

################################################################################

# To Create data frame

retailsales.data <- read.csv("RetailSalesTSA.csv")
head(retailsales.data)   # determine beginning of the dataset
tail(retailsales.data)   # determine ending of the dataset

# Creating a time series of gold.ts in R using ts() function 
retailsales.ts <- ts(retailsales.data$Sales, 
                     start = c(1995,1), end = c(2021,12), freq = 12)
head(retailsales.ts)
tail(retailsales.ts)
retailsales.ts


###################Plotting the time series data and visualizing.###########

plot(retailsales.ts, xlim=c(1995,2024),ylim=c(0,7000),
     xlab = "Years", ylab = "Retail Sales (In $Millions)", bty = "l",
     xaxt = "n", main = "Retail Sales : Hobby and Game Stores", lwd = 1) 
axis(1, at = seq(1995, 2024, 1), labels = format(seq(1995, 2024, 1)))
points(retailsales.ts,col=rainbow(12),pch=20,cex = 0.7)
lines(retailsales.ts, col = "black", lty = 1, lwd = 1)
legend(1995,6500,legend = c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"),
      col=rainbow(12),border = "black",box.lwd = 1,pch =20,title = "Months of the year",cex = 0.5,horiz = TRUE)

boxplot(retailsales.ts ~ cycle(retailsales.ts),vertical=TRUE,col='lightblue',ylab = "Retail Sales (In $Millions)",xlab = "Months",
        main = "Box Plot of Retail(Monthly) Sales",axes=TRUE,outline=FALSE)

autoplot(retailsales.ts,col='darkblue',ylab = "Retail Sales (In $Millions)",main="Time Series components of Retail Sales")

Acf(retailsales.ts,lag.max = 12,type = c("correlation", "covariance", "partial"),main="AutoCorrelation Plot For Retail Sales")

#########################STL Function######################################

# Season, Trend and Linearity stl() time series components

sales.stl <- stl(retailsales.ts, s.window = "periodic")
sales.stl
autoplot(sales.stl,main = "Retail Sales Time Series Components")


################################################################################

# Training and Validation Partition

nValid <- round(length(retailsales.ts) * (0.26))
nValid
nTrain <- (length(retailsales.ts) - nValid)
nTrain

train.ts <- window(retailsales.ts, start = c(1995, 1), end = c(1995, nTrain))
valid.ts <- window(retailsales.ts, start = c(1995, nTrain + 1), 
                   end = c(1995, nTrain + nValid))

train.ts
valid.ts

# Plot the time series data and visualize partitions. 
plot(train.ts, 
     xlab = "Time", ylab = "Retail Sales (in millions)", ylim = c(300, 6000), bty = "l",
     xaxt = "n", xlim = c(1995, 2021), 
     main = "Retail Sales (Game stores): Training and Validation Partitions", lwd = 1) 
axis(1, at = seq(1995, 2021, 1), labels = format(seq(1995, 2021, 1)))
lines(valid.ts, col = "red", lty = 1, lwd = 1)

legend(1995,6000, legend = c("Training Data", 
                             "Validation Data"), 
       col = c("black", "red"), 
       lty = 1, lwd =1, bty = "n")

# Horizontal lines
text(2006, 4500, "Training")
text(2018.5, 4500, "Validation")
arrows(1995, 4200, 2015, 4200, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2015, 4200, 2022, 4200, code = 3, length = 0.1,
       lwd = 1, angle = 30)

# Vertical lines
abline(v=1995)
abline(v=2015)
abline(v=2022)

################################################################################
##--------------------------Predictability test -------------------------

summary(Arima(retailsales.ts,order = c(1,0,0)))

auto.arima(retailsales.ts, max.p=1, max.q=0, max.d = 0, stepwise=FALSE, approximation=FALSE)

pnorm((0.3024 - 1)/(0.0508))

Acf(diff(retailsales.ts,lag = 1),lag.max = 12,main = "Autocorrelation Plot for First difference retail Sales")

################################################################################
##--------------------------Z-test (null-Hypothesis)-------------------------
retail.ar1<- Arima(retailsales.ts, order = c(1,0,0))
summary(retail.ar1)
# Apply z-test to test the null hypothesis that beta 
# coefficient of AR(1) is equal to 1.
ar1 <- 0.3024
s.e. <- 0.0508                     # s.e.: standard error of estimate
null_mean <- 1                     # Ho: Beta =1
alpha <- 0.01                      #confidence level..consider it as 0.05 or 0.01
z.stat <- (ar1-null_mean)/s.e.     # ztest statistic
z.stat
p.value <- pnorm(z.stat)
p.value                            # Thus, beta =1 and accepts null hypothesis so it is random walk
if (p.value<alpha) {
  "Reject null hypothesis"
} else {
  "Accept null hypothesis"
}
################################################################################
# For AR(1) model, creating lag-1 differencing
# Creating first differencing data using lag1.

diff.retail.ts <- diff(retailsales.ts, lag = 1)
diff.retail.ts

# Using Acf() function to identify autocorrelation for first differencing 
# retail sales, and plot autocorrelation for different lags 
# (up to maximum of 12).
Acf(diff.retail.ts, lag.max = 12, 
    main = "Autocorrelation for 1st Differencing Retail Sales: Game stores")

### It can be observed that for lag 1, lag2, lag 10, lag 11 and lag 12, autocorrelation coefficients have significance. 
# This plot indicates the correlation coefficients at lag 1, lag2, lag 10, lag 11 and lag 12 are not within horizontal thresholds, 
# hence it can be inferred that Retail sales data is not random walk, can be predicted.
# Since lag 12 have much more significant value, its an evidence of presence of seasonality 


################################################################################
## MODEL 1: REGRESSION MODELS ##
################################################################################

#### FOR Training Partition and forecast on validation ####

############### Regression model with linear trend ###############

#Use the tslm() function for the training partition to develop Regression model with linear trend
train.linear <- tslm(train.ts ~ trend)

# Applying the summary () function to identify the model structure and parameters for regression model
summary(train.linear)

# forecasting validation period using the forecast() function.
train.linear.for <- forecast(train.linear, h = nValid, level = 0)
train.linear.for

################### Regression model with quadratic trend ######################

#  Use the tslm() function for the training partition to develop Regression model with quadratic trend
train.quadratic <- tslm(train.ts ~ trend + I(trend^2))

# Applying the summary () function to identify the model structure and parameters for regression model
summary(train.quadratic)

# forecasting validation period using the forecast() function.
train.quadratic.for <- forecast(train.quadratic, h = nValid, level = 0)
train.quadratic.for


##################### Regression model with seasonality ########################

# Use the tslm() function for the training partition to develop Regression model with seasonality 
train.seasonality <- tslm(train.ts ~ season)

# Applying the summary () function to identify the model structure and parameters for regression model
summary(train.seasonality)

# forecasting validation period using the forecast() function.
train.seasonality.for <- forecast(train.seasonality, h = nValid, level = 0)
train.seasonality.for


############ Regression model with linear trend and seasonality ################

# Use the tslm() function for the training partition to 
# develop Regression model with linear trend and seasonality 
train.linear.seasonality <- tslm(train.ts ~ trend + season)

# Applying the summary () function to identify the model structure and parameters for regression model
summary(train.linear.seasonality)

# forecasting validation period using the forecast() function.
train.linear.seasonality.for <- forecast(train.linear.seasonality, h = nValid, level = 0)
train.linear.seasonality.for


############## Regression model with quadratic trend and seasonality ##########

# Use the tslm() function for the training partition to 
# develop Regression model with quadratic trend and seasonality 
train.quadratic.seasonality <- tslm(train.ts ~ trend + I(trend^2) + season)

# Applying the summary () function to identify the model structure and parameters for regression model
summary(train.quadratic.seasonality)

# forecasting validation period using the forecast() function.
train.quadratic.seasonality.for <- forecast(train.quadratic.seasonality, h = nValid, level = 0)
train.quadratic.seasonality.for

### In terms of R2 and statistics, Regression model with quadratic trend and seasonality 
# seems like a best model for training partition 

# Apply the accuracy () function to compare performance measure of the 5 forecasts

# Accuracy for linear trend
round(accuracy(train.linear.for$mean, valid.ts),3)


# Accuracy for quadratic trend
round(accuracy(train.quadratic.for$mean, valid.ts),3)


# Accuracy for seasonality
round(accuracy(train.seasonality.for$mean, valid.ts),3)


# Accuracy for linear trend and seasonality
round(accuracy(train.linear.seasonality.for$mean, valid.ts),3)


# Accuracy for quadratic trend seasonality
round(accuracy(train.quadratic.seasonality.for$mean, valid.ts),3)


linear_trend_train_RMSE <- round(accuracy(train.linear.for$mean, valid.ts),3)[2]
linear_trend_train_MAPE <- round(accuracy(train.linear.for$mean, valid.ts),3)[5]

quadratic_trend_train_RMSE <- round(accuracy(train.quadratic.for$mean, valid.ts),3)[2]
quadratic_trend_train_MAPE <- round(accuracy(train.quadratic.for$mean, valid.ts),3)[5]

seasonality_train_RMSE <- round(accuracy(train.seasonality.for$mean, valid.ts),3)[2]
seasonality_train_MAPE <- round(accuracy(train.seasonality.for$mean, valid.ts),3)[5]

linear_trend_seasonality_train_RMSE <- round(accuracy(train.linear.seasonality.for$mean, valid.ts),3)[2]
linear_trend_seasonality_train_MAPE <- round(accuracy(train.linear.seasonality.for$mean, valid.ts),3)[5]

quadratic_trend_seasonality_train_RMSE <- round(accuracy(train.quadratic.seasonality.for$mean, valid.ts),3)[2]
quadratic_trend_seasonality_train_MAPE <- round(accuracy(train.quadratic.seasonality.for$mean, valid.ts),3)[5]


### In terms of accuracy, linear trend and seasonality is best model for training partition

################################################################################
## Plot Accuracy measures of all models in training partition
## Plot for RMSE

# Simple Bar Plot
# Load ggplot2
library(ggplot2)

# For RMSE
rmse_data <- data.frame(
  Model=c("Linear Trend", "Quadratic Trend", "Seasonality", "Linear Trend & Seasonality", "Quadratic Trend & Seasonality") ,  
  RMSE=c(linear_trend_train_RMSE,quadratic_trend_train_RMSE,seasonality_train_RMSE,linear_trend_seasonality_train_RMSE,quadratic_trend_seasonality_train_RMSE)
)

ggplot(rmse_data, aes(x=Model, y=RMSE)) + ggtitle("Regression Models: Training & Validation Partitions RMSE Accuracy Measures") +
  geom_bar(stat = "identity", fill ="steelblue",width=0.4) + 
  geom_text(aes(label=RMSE), vjust=1.6, color="white", size=3.5) + 
  theme_minimal()

# For MAPE
mape_data <- data.frame(
  Model=c("Linear Trend", "Quadratic Trend", "Seasonality", "Linear Trend & Seasonality", "Quadratic Trend & Seasonality") ,  
  MAPE=c(linear_trend_train_MAPE,quadratic_trend_train_MAPE,seasonality_train_MAPE,linear_trend_seasonality_train_MAPE,quadratic_trend_seasonality_train_MAPE)
)


ggplot(mape_data, aes(x=Model, y=MAPE)) + ggtitle("Regression Models: Training & Validation Partitions MAPE Accuracy Measures") +
  geom_bar(stat = "identity", fill ="#FF9950",width=0.4) + 
  geom_text(aes(label=MAPE), vjust=1.6, color="white", size=3.5) + 
  theme_minimal()

##########################

#### To create Forecast for 24 months of year 2022 and 2023 ####

####### Using the entire data set to develop Regression model with linear trend and seasonality 
linear.trend.seasonality <- tslm(retailsales.ts ~ trend + season)
summary(linear.trend.seasonality)
linear.trend.seasonality.for <- forecast(linear.trend.seasonality, h = 24, level = 95)
linear.trend.seasonality.for

###### Using the entire data set to develop Regression model with quadratic trend and seasonality 
quadratic.trend.seasonality <- tslm(retailsales.ts ~ trend + I(trend^2) + season)
summary(quadratic.trend.seasonality)
quadratic.trend.seasonality.for <- forecast(quadratic.trend.seasonality, h = 24, level = 95)
quadratic.trend.seasonality.for

###### Apply the accuracy () function to compare performance measure of the 5 forecasts


# Accuracy for linear trend and seasonality
round(accuracy(linear.trend.seasonality.for$fitted, retailsales.ts), 3)

# Accuracy for quadratic trend and Seasonality
round(accuracy(quadratic.trend.seasonality.for$fitted, retailsales.ts), 3)


#naive_entire_RMSE <- round(accuracy((naive(retailsales.ts))$fitted, retailsales.ts), 3)[2]
#naive_entire_MAPE <- round(accuracy((naive(retailsales.ts))$fitted, retailsales.ts), 3)[5]

#snaive_entire_RMSE <- round(accuracy((snaive(retailsales.ts))$fitted, retailsales.ts), 3)[2]
#snaive_entire_MAPE <- round(accuracy((snaive(retailsales.ts))$fitted, retailsales.ts), 3)[5]


linear_trend_seasonality_entire_RMSE <- round(accuracy(linear.trend.seasonality.for$fitted, retailsales.ts), 3)[2]
linear_trend_seasonality_entire_MAPE <- round(accuracy(linear.trend.seasonality.for$fitted, retailsales.ts), 3)[5]

quadratic_trend_seasonality_entire_RMSE <- round(accuracy(quadratic.trend.seasonality.for$fitted, retailsales.ts), 3)[2]
quadratic_trend_seasonality_entire_MAPE <- round(accuracy(quadratic.trend.seasonality.for$fitted, retailsales.ts), 3)[5]

## Plot Accuracy measures of all models in entire dataset
# For RMSE
rmse_entire_data <- data.frame(
  Model=c("Linear Trend & Seasonality", "Quadratic Trend & Seasonality") ,  
  RMSE=c(linear_trend_seasonality_entire_RMSE,quadratic_trend_seasonality_entire_RMSE)
)

ggplot(rmse_entire_data, aes(x=Model, y=RMSE)) + ggtitle("Regression Models: Entire dataset RMSE Accuracy Measures") +
  geom_bar(stat = "identity", fill ="steelblue",width=0.2) + 
  geom_text(aes(label=RMSE), vjust=1.6, color="white", size=3.5) + 
  theme_minimal()

# For MAPE
mape_entire_data <- data.frame(
  Model=c("Linear Trend & Seasonality", "Quadratic Trend & Seasonality") ,  
  MAPE=c(linear_trend_seasonality_entire_MAPE, quadratic_trend_seasonality_entire_MAPE)
)


ggplot(mape_entire_data, aes(x=Model, y=MAPE)) + ggtitle("Regression Models: Entire dataset MAPE Accuracy Measures") +
  geom_bar(stat = "identity", fill ="#FF9950",width=0.2) + 
  geom_text(aes(label=MAPE), vjust=1.6, color="white", size=3.5) + 
  theme_minimal()

############################################

# Plotting Future Predictions
# plot HW predictions for original data, optimal smoothing parameters.
plot(quadratic.trend.seasonality.for$mean, 
     xlab = "Time", ylab = "Retail Sales (in millions)", ylim = c(300, 6000), bty = "l",
     xaxt = "n", xlim = c(1995, 2024),
     main = "QTS Model for Entire Data Set and Forecast for future 24 Periods", 
     lty = 2, col = "orange", lwd = 2) 
axis(1, at = seq(1995, 2024, 1), labels = format(seq(1995, 2024, 1)))
lines(quadratic.trend.seasonality.for$fitted, col = "lightblue", lwd = 2)
lines(retailsales.ts)

legend(1995,6000, 
       legend = c("Retail Sales Entire Dataset", 
                  "QTS Model for Entire Data Set",
                  "QTS Model's Forecast, future 24 Periods"), 
       col = c("black", "lightblue" , "orange"), 
       lty = c(1, 1, 2), lwd =c(1, 2, 2), bty = "n")


# Horizontal lines
text(2009, 4500, "Entire Dataset")
text(2023, 4500, "Future")
arrows(1995, 4200, 2022, 4200, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2022, 4200, 2024, 4200, code = 3, length = 0.1,
       lwd = 1, angle = 30)

# Vertical lines
abline(v=1995)
abline(v=2022)
abline(v=2024)

###################################################################################

###################################################################################
## MODEL 2: Holt Winter's Model - Automated ZZZ selection of Model Options (ZZZ) ##
###################################################################################

# Create Holt-Winter's (HW) exponential smoothing for partitioned data. Use ets() function with model = "ZZZ", i.e., automated selection 
# error, trend, and seasonality options.Use optimal alpha, beta, & gamma to fit HW over the training period.
hw.ZZZ <- ets(train.ts, model = "ZZZ")
hw.ZZZ

# Use forecast() function to make predictions using this HW model with validation period (nValid). 
# Show predictions in tabular format.
hw.ZZZ.pred <- forecast(hw.ZZZ, h = nValid, level = 0)
hw.ZZZ.pred

# Plot HW predictions for original data, automated selection of the 
# model and optimal smoothing parameters.
plot(hw.ZZZ.pred$mean, 
     xlab = "Time", ylab = "Retail Sales (in millions)", ylim = c(300, 6000), bty = "l",
     xaxt = "n", xlim = c(1995, 2024), 
     main = "Holt-Winter's Model with Automated Selection of Model Options", lty = 5, col = "orange", lwd = 2) 
axis(1, at = seq(1995, 2021, 1), labels = format(seq(1995, 2021, 1)))
lines(hw.ZZZ.pred$fitted, col = "lightblue", lwd = 2)
lines(retailsales.ts)

legend(1995,6000, 
       legend = c("Retail Sales", 
                  "Holt-Winter's Automated Model for Training Partition",
                  "Holt-Winter's Automated Model for Validation Partition"), 
       col = c("black", "lightblue" , "orange"), 
       lty = c(1, 1, 2), lwd =c(1, 2, 2), bty = "n")

# Horizontal lines
text(2006, 4500, "Training")
text(2018, 4500, "Validation")
text(2023, 4500, "Future")
arrows(1995, 4200, 2015, 4200, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2015, 4200, 2022, 4200, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2022, 4200, 2024, 4200, code = 3, length = 0.1,
       lwd = 1, angle = 30)

# Vertical lines
abline(v=1995)
abline(v=2015)
abline(v=2022)

######################

#Comparing Accuracy of HW model &  for training partition

# For QTS Regression
# Already developed in Model 1
quadratic_trend_seasonality_train_RMSE 
quadratic_trend_seasonality_train_MAPE 


# For HW model
# Accuracy for quadratic trend seasonality for training partiton
round(accuracy(hw.ZZZ.pred$mean, valid.ts), 3)

hw_train_RMSE <- round(accuracy(hw.ZZZ.pred$mean, valid.ts), 3)[2]
hw_train_MAPE <- round(accuracy(hw.ZZZ.pred$mean, valid.ts), 3)[5]

## Plot Accuracy measures HW and QTS regression model in training data set
# Simple Bar Plot
# Load ggplot2
library(ggplot2)

# For RMSE
hw_rmse_train_data <- data.frame(
  Model=c("Quadratic Trend & Seasonality","Holt Winter's") ,  
  RMSE=c(quadratic_trend_seasonality_train_RMSE, hw_train_RMSE)
)

ggplot(hw_rmse_train_data, aes(x=Model, y=RMSE)) + ggtitle("Training Partition: Accuracy measure RMSE, HW vs Model 1") +
  geom_bar(stat = "identity", fill ="steelblue",width=0.2) + 
  geom_text(aes(label=RMSE), vjust=1.6, color="white", size=3.5) + 
  theme_minimal()

# For MAPE
hw_mape_entire_data <- data.frame(
  Model=c("Quadratic Trend & Seasonality","Holt Winter's"),
  MAPE=c(quadratic_trend_seasonality_train_MAPE, hw_train_MAPE)
)


ggplot(hw_mape_entire_data, aes(x=Model, y=MAPE)) + ggtitle("Training Partition: Accuracy measure MAPE, HW vs Model 1") +
  geom_bar(stat = "identity", fill ="#FF9950",width=0.2) + 
  geom_text(aes(label=MAPE), vjust=1.6, color="white", size=3.5) + 
  theme_minimal()

##################################################################################

# For Entire Dataset

# Create Holt-Winter's (HW) exponential smoothing for full Retail data set. 
# Use ets() function with model = "ZZZ", to identify the best HW option
# and optimal alpha, beta, & gamma to fit HW for the entire data period.
HW.ZZZ <- ets(retailsales.ts, model = "ZZZ")
HW.ZZZ 

# Using forecast() function to make predictions using this HW model for
# 12 months into the future.
HW.ZZZ.pred <- forecast(HW.ZZZ, h = 24 , level = 95)
HW.ZZZ.pred

################################################################################

#Comparing Accuracy of HW model & quad trend seasonality model for Entire dataset
round(accuracy(HW.ZZZ.pred$fitted, retailsales.ts), 3)

#Comparing Accuracy of HW model & QTS for entire dataset
# For QTS Regression
# Already developed in Model 1
quadratic_trend_seasonality_entire_RMSE 
quadratic_trend_seasonality_entire_MAPE 

# for HW model
hw_entire_RMSE <- round(accuracy(HW.ZZZ.pred$fitted, retailsales.ts), 3)[2]
hw_entire_MAPE <- round(accuracy(HW.ZZZ.pred$fitted, retailsales.ts), 3)[5]

## Plot Accuracy measures HW and QTS regression model in training data set
# Simple Bar Plot
# Load ggplot2
library(ggplot2)

# For RMSE
hw_rmse_entire_data <- data.frame(
  Model=c("Quadratic Trend & Seasonality","Holt Winter's") ,  
  RMSE=c(quadratic_trend_seasonality_entire_RMSE, hw_entire_RMSE)
)

ggplot(hw_rmse_entire_data, aes(x=Model, y=RMSE)) + ggtitle("Entire Dataset: Accuracy measure RMSE, HW vs Model 1") +
  geom_bar(stat = "identity", fill ="steelblue",width=0.2) + 
  geom_text(aes(label=RMSE), vjust=1.6, color="white", size=3.5) + 
  theme_minimal()

# For MAPE
hw_mape_entire_data <- data.frame(
  Model=c("Quadratic Trend & Seasonality","Holt Winter's"),
  MAPE=c(quadratic_trend_seasonality_entire_MAPE, hw_entire_MAPE)
)


ggplot(hw_mape_entire_data, aes(x=Model, y=MAPE)) + ggtitle("Entire Dataset: Accuracy measure MAPE, HW vs Model 1") +
  geom_bar(stat = "identity", fill ="#FF9950",width=0.2) + 
  geom_text(aes(label=MAPE), vjust=1.6, color="white", size=3.5) + 
  theme_minimal()

################################################################################

# Plotting Future Predictions
# plot HW predictions for original data, optimal smoothing parameters.
plot(HW.ZZZ.pred$mean, 
     xlab = "Time", ylab = "Retail Sales (in millions)", ylim = c(300, 6000), bty = "l",
     xaxt = "n", xlim = c(1995, 2024),
     main = "Holt-Winter's Automated Model for Entire Data Set and Forecast for future 24 Periods", 
     lty = 2, col = "orange", lwd = 2) 
axis(1, at = seq(1995, 2024, 1), labels = format(seq(1995, 2024, 1)))
lines(HW.ZZZ.pred$fitted, col = "lightblue", lwd = 2)
lines(retailsales.ts)

legend(1995,6000, 
       legend = c("Retail Sales Entire Dataset", 
                  "Holt-Winter's Automated Model for Entire Data Set",
                  "Holt-Winter's Automated Model's Forecast, future 24 Periods"), 
       col = c("black", "lightblue" , "orange"), 
       lty = c(1, 1, 2), lwd =c(1, 2, 2), bty = "n")


# Horizontal lines
text(2009, 4500, "Entire Dataset")
text(2023, 4500, "Future")
arrows(1995, 4200, 2022, 4200, code = 3, length = 0.1,
       lwd = 1, angle = 30)

# Vertical lines
abline(v=1995)
abline(v=2022)
abline(v=2024)
################################################################################

################################################################################
### TWO LEVEL FORECAST - Regression with Quadratic trend and Seasonality, AR(12) for residuals
################################################################################


# for training dataset
train.quadratic.seasonality
train.quadratic.seasonality.for

# Use Arima() function to fit AR(1) model for regression residuals.
# The ARIMA model order of order = c(1,0,0) gives an AR(12) model.
# Use forecast() function to make prediction of residuals into the future nvalid months.
residual.ar1 <- Arima(train.quadratic.seasonality$residuals, order = c(12,0,0))
residual.ar1.pred <- forecast(residual.ar1, h = nValid, level = 0)
residual.ar1 # et = -0.0791 + 0.2919 et-1
residual.ar1.pred

# Use Acf() function to identify autocorrealtion for the residual of residuals 
# and plot autocorrelation for different lags (up to maximum of 12).
Acf(residual.ar1$residuals, lag.max = 12, 
    main = "Autocorrelation for Retail Residuals of Residuals for Training Data Set")

# Identify forecast for the future 24 periods as sum of quadratic trend and seasonal model
# and AR(1) model for residuals.
quad.season.ar1.pred <- train.quadratic.seasonality.for$mean + residual.ar1.pred$mean
quad.season.ar1.pred

# Create a data table with quadratic trend and seasonal forecast for 24 future periods,
# AR(1) model for residuals for 24 future periods, and combined two-level forecast for
# 24 future periods. 
valid.df <- data.frame(train.quadratic.seasonality.for$mean, 
                       residual.ar1.pred$mean, quad.season.ar1.pred, valid.ts)
names(valid.df) <- c("Reg.Forecast", "AR(1)Forecast","Combined.Forecast","Original data")
valid.df

# accuracy
round(accuracy(quad.season.ar1.pred, valid.ts), 3)

################################## For Entire Dataset###########################

###### Using the entire data set to develop Regression model with quadratic trend and seasonality 
#from above code
quadratic.trend.seasonality.for

# Use Arima() function to fit AR(1) model for regression residuals.
# The ARIMA model order of order = c(1,0,0) gives an AR(12) model.
# Use forecast() function to make prediction of residuals into the future nvalid months.
residual.ar1.for <- Arima(quadratic.trend.seasonality.for$residuals, order = c(12,0,0))
residual.ar1.pred <- forecast(residual.ar1.for, h = 24, level = 0)

residual.ar1.for
# Use Acf() function to identify autocorrealtion for the residual of residuals 
# and plot autocorrelation for different lags (up to maximum of 12).
Acf(residual.ar1.for$residuals, lag.max = 12, 
    main = "Autocorrelation for Retail Residuals of Residuals for Enitre data Set")


# Identify forecast for the future 24 periods as sum of quadratic trend and seasonal model
# and AR(1) model for residuals.
trend.season.ar1.pred <- quadratic.trend.seasonality.for$mean + residual.ar1.pred$mean
trend.season.ar1.pred


# Create a data table with quadratic trend and seasonal forecast for 12 future periods,
# AR(1) model for residuals for 12 future periods, and combined two-level forecast for
# 12 future periods. 
table.df <- data.frame(quadratic.trend.seasonality.for$mean, 
                       residual.ar1.pred$mean, trend.season.ar1.pred)
names(table.df) <- c("Reg.Forecast", "AR(1)Forecast","Combined.Forecast")
table.df

round(accuracy(quadratic.trend.seasonality$fitted + residual.ar1$fitted, retailsales.ts), 3)

### PLOT 1 for training
plot(train.quadratic.seasonality.for$mean, 
     xlab = "Time", ylab = "Retail Sales (in millions)", ylim = c(300, 6000), bty = "l",
     xaxt = "n", xlim = c(1995, 2024), 
     main = "Two Level Model with Regression with Quadratic and seasonal", lty = 5, col = "blue", lwd = 2) 
axis(1, at = seq(1995, 2021, 1), labels = format(seq(1995, 2021, 1)))
lines(train.quadratic.seasonality.for$fitted, col = "blue", lwd = 2)
lines(retailsales.ts)

legend(1995,6000, 
       legend = c("Retail Sales", 
                  "QTS Model for Training Partition",
                  "QTS Model for Validation Partition"), 
       col = c("black", "blue" , "blue"), 
       lty = c(1, 1, 2), lwd =c(1, 2, 2), bty = "n")
# Horizontal lines
text(2006, 4500, "Training")
text(2018, 4500, "Validation")
text(2023, 4500, "Future")
arrows(1995, 4200, 2015, 4200, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2015, 4200, 2022, 4200, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2022, 4200, 2024, 4200, code = 3, length = 0.1,
       lwd = 1, angle = 30)

# Vertical lines
abline(v=1995)
abline(v=2015)
abline(v=2022)

## Plot 2 for entire data set
# Plotting Future Predictions
# plot HW predictions for original data, optimal smoothing parameters.
plot(trend.season.ar1.pred, 
     xlab = "Time", ylab = "Retail Sales (in millions)", ylim = c(300, 6000), bty = "l",
     xaxt = "n", xlim = c(1995, 2024),
     main = "Two level forecast Model for Entire Data Set and Forecast for Future 24 Periods", 
     lty = 2, col = "red", lwd = 2) 
axis(1, at = seq(1995, 2024, 1), labels = format(seq(1995, 2024, 1)))
lines(quadratic.trend.seasonality$fitted, col = "blue", lwd = 2)
lines(retailsales.ts)

legend(1995,6000, 
       legend = c("Retail Sales", 
                  "QTS Model for Entire Data Set",
                  "Two level Model Forecast, Future 24 Periods"), 
       col = c("black", "blue" , "red"), 
       lty = c(1, 1, 2), lwd =c(1, 2, 2), bty = "n")


# Horizontal lines
text(2009, 4500, "Entire Dataset")
text(2023, 4500, "Future")
arrows(1995, 4200, 2022, 4200, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2022, 4200, 2024, 4200, code = 3, length = 0.1,
       lwd = 1, angle = 30)

# Vertical lines
abline(v=1995)
abline(v=2022)
abline(v=2024)


################################################################################

##---------------- FIT ARIMA(2,1,2)(1,1,2) MODEL for training data--------------

################################################################################
train.arima.seas <- Arima(train.ts, order = c(2,1,2), 
                          seasonal = c(1,1,2)) 
summary(train.arima.seas)

# yt - yt-1 =  - 0.604  (yt-1 -yt-2) + 0.271  (yt-2 -yt-3) + 0.206  et-1 – 0.692 et-2 - 0.210  (yt-1 -yt-13) - 0.436 rt-1  – 0.233 rt-2

#summary(Arima(retailsales.ts,c(2,0,0))) #AR(2)
#summary(Arima(retailsales.ts,c(0,0,1))) # MA(1)

train.arima.seas.pred <- forecast(train.arima.seas, h = nValid, level = 0)
train.arima.seas.pred

# Use Acf() function to create autocorrelation chart of ARIMA(2,1,2)(1,1,2) 
# model residuals.
Acf(train.arima.seas$residuals, lag.max = 12, 
    main = "Autocorrelations of ARIMA(2,1,2)(1,1,2) Model Residuals")

# Plot ts data, ARIMA model, and predictions for validation period.
plot(train.arima.seas.pred, 
     xlab = "Time", ylab = "Retail Sales (in $Millions)", ylim = c(500, 7000), bty = "l",
     xaxt = "n", xlim = c(1995, 2024), 
     main = "ARIMA(2,1,2)(1,1,2)[12] Model", lwd = 2, flty = 5) 
axis(1, at = seq(1995, 2024, 1), labels = format(seq(1995, 2024, 1)))
lines(train.arima.seas.pred$fitted, col = "blue", lwd = 1)
lines(valid.ts, col = "black", lwd = 1, lty = 1)
legend(1995,7000, legend = c("Retail Sales (in $Millions)", 
                             "Seasonal ARIMA Forecast for Training Period",
                             "Seasonal ARIMA Forecast for Validation Period"), 
       col = c("black", "blue" , "lightblue"), 
       lty = c(1, 1, 5), lwd =c(2, 2, 2), bty = "n")

# Plot on the chart vertical lines and horizontal arrows
# describing training, validation, and future prediction intervals.
lines(c(2015, 2015), c(0, 5500))
lines(c(2022, 2022), c(0, 5500))
text(2001, 5000, "Training")
text(2019, 5000, "Validation")
text(2023.75, 5000, "Future")
arrows(1995, 4750, 2015, 4750, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2015, 4750, 2022, 4750, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2022, 4750, 2024.75, 4750, code = 3, length = 0.1,
       lwd = 1, angle = 30)


################################################################################
## -------------------FIT AUTO ARIMA MODEL for training data.-------------------
################################################################################
train.auto.arima <- auto.arima(train.ts)
summary(train.auto.arima)

# yt - yt-1 =  0.322  (yt-1 -yt-2) - 0.745  et-1- 0.702 rt-1

train.auto.arima.pred <- forecast(train.auto.arima, h = nValid, level = 0)
train.auto.arima.pred

# Using Acf() function, create autocorrelation chart of auto ARIMA 
# model residuals.
Acf(train.auto.arima$residuals, lag.max = 12, 
    main = "Autocorrelations of Auto ARIMA Model Residuals")

# Plot ts data, trend and seasonality data, and predictions for validation period.
plot(train.auto.arima.pred, 
     xlab = "Time", ylab = "Prices", ylim = c(500, 7000), bty = "l",
     xaxt = "n", xlim = c(1995, 2024.5), 
     main = "Auto ARIMA Model", lwd = 1, flty = 5) 
axis(1, at = seq(1995, 2024, 1), labels = format(seq(1995, 2024, 1)))
lines(train.auto.arima.pred$fitted, col = "blue", lwd = 1)
lines(valid.ts, col = "black", lwd = 1, lty = 1)
legend(1995,7000, legend = c("Retail Sales (in $Millions)", 
                             "Auto ARIMA Forecast for Training Period",
                             "Auto ARIMA Forecast for Validation Period"), 
       col = c("black", "blue" , "lightblue"), 
       lty = c(1, 1, 5), lwd =c(2, 2, 2), bty = "n")

# Plot on the chart vertical lines and horizontal arrows
# describing training, validation, and future prediction intervals.
lines(c(2015, 2015), c(0, 5500))
lines(c(2022, 2022), c(0, 5500))
text(2001, 5000, "Training")
text(2019, 5000, "Validation")
text(2023.75, 5000, "Future")
arrows(1995, 4750, 2015, 4750, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2015, 4750, 2022, 4750, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2022, 4750, 2024.75, 4750, code = 3, length = 0.1,
       lwd = 1, angle = 30)


##---------- SEASONAL ARIMA AND AUTO ARIMA MODELS FOR ENTIRE DATA SET.----------
#SEASONAL ARIMA
arima.seas <- Arima(retailsales.ts, order = c(2,1,2), 
                    seasonal = c(1,1,2)) 
summary(arima.seas)

# seasonal ARIMA model for the future 24 periods. 
arima.seas.pred <- forecast(arima.seas, h = 24, level = c(80,95))
arima.seas.pred

# autocorrelation chart of seasonal ARIMA 

Acf(arima.seas$residuals, lag.max = 12, 
    main = "Autocorrelations of Seasonal ARIMA (2,1,2)(1,1,2) Model Residuals")


plot(retailsales.ts, 
     xlab = "Time", ylab = "Retail Sales (in $Millions)", ylim = c(500, 6000), bty = "l",
     xaxt = "n", xlim = c(1995, 2024.5), lwd = 1,
     main = "Seasonal ARIMA(2,1,2)(1,1,2)[12] Model") 
axis(1, at = seq(1995, 2024, 1), labels = format(seq(1995, 2024, 1)))
lines(arima.seas$fitted, col = "blue", lwd = 1)
lines(arima.seas.pred$mean, col = "orange", lty = 5, lwd = 1)
legend(1995,7000, legend = c("Retail Sales (in $Millions)", 
                             "Seasonal ARIMA Forecast for Training Period",
                             "Seasonal ARIMA Forecast for Future"), 
       col = c("black", "blue" , "orange"), 
       lty = c(1, 1, 5), lwd =c(2, 2, 2), bty = "n")

lines(c(2022, 2022), c(0, 5750))
text(2008.5, 5000, "Training")
text(2023.75, 5000, "Future")
arrows(1995, 4750, 2022, 4750, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2022, 4750, 2025.5, 4750, code = 3, length = 0.1,
       lwd = 1, angle = 30)

#with confidence intervals 

plot(arima.seas.pred, 
     xlab = "Time", ylab = "Retail Sales (in $millions)", ylim = c(0,7000), bty = "l",
     xlim = c(2016, 2024), main = "80% and 95% Confidence Intervals of Auto ARIMA(2,1,1)(0,1,2)[12] \nForecast For future 24 periods"
     ,col='darkblue',lwd=2) 
axis(1, at = seq(2016, 2024, 1), labels = format(seq(2016, 2024, 1)) )

legend(2016.25,7000, legend = c("Retail Sales Data", "Point Forecast for Future 24 periods",
                                "95% Confidence Interval", 
                                "80% Confidence Interval"),
       col = c("blue","lightblue", "lightgrey", "grey"), 
       lty = c(1,1,1,1),lwd =c(2,2,5,5), bty = "n")

lines(c(2022, 2022), c(0,6000))
text(2023.25, 4950, "Future 24 Periods")
arrows(2022, 4700, 2024, 4700, code = 3, length = 0.1,
       lwd = 1, angle = 30)



#############AUTO ARIMA for full data ##########################################

auto.arima.full <- auto.arima(retailsales.ts)
summary(auto.arima.full)

auto.arima.full.pred <- forecast(auto.arima.full,h = 24,level = c(80,95))
auto.arima.full.pred$mean

plot(retailsales.ts, 
     xlab = "Time", ylab = "Retail Sales (in $millions)", ylim = c(500,6000), bty = "l",
     xaxt = "n", xlim = c(1995, 2025), lwd = 1,
     main = "Auto ARIMA Model for Entire Dataset") 
axis(1, at = seq(1995, 2025, 2), labels = format(seq(1995, 2025, 2)) )
lines(auto.arima.full.pred$mean,col='red',lwd=1,lty=2)

legend(1995,6000, legend = c("Retail Sales Data", 
                             "Auto ARIMA Forecast for future 24 periods"),
       col = c("blue", "red"), 
       lty = c(1,2),lwd =c(1,2), bty = "n")

lines(c(2022, 2022), c(0,6000))
text(2008.25, 4750, "Training")
text(2023.25,4750, "Future")
arrows(2022,4500, 1995,4500, code = 3, length = 0.1,
       lwd = 1, angle = 30)
arrows(2022, 4500, 2025, 4500, code = 3, length = 0.1,
       lwd = 1, angle = 30)

#Confidence interval

plot(auto.arima.full.pred, 
     xlab = "Time", ylab = "Retail Sales (in $millions)", ylim = c(0,7000), bty = "l",
     xlim = c(2016, 2024), main = "80% and 95% Confidence Intervals of Auto ARIMA(2,1,1)(0,1,2)[12] \nForecast For future 24 periods"
     ,col='darkblue',lwd=2) 
axis(1, at = seq(2016, 2024, 1), labels = format(seq(2016, 2024, 1)) )

legend(2018.25,7000, legend = c("Retail Sales Data", "Point Forecast for Future 24 periods",
                                "95% Confidence Interval", 
                                "80% Confidence Interval"),
       col = c("blue","lightblue", "lightgrey", "grey"), 
       lty = c(1,1,1,1),lwd =c(2,2,5,5), bty = "n")

lines(c(2022, 2022), c(0,5000))
text(2023.25, 4750, "Future 24 Periods")
arrows(2022, 4500, 2024, 4500, code = 3, length = 0.1,
       lwd = 1, angle = 30)


#################### ALL ACCURACY COMPARISON ###################################

# Accuracy for linear trend and seasonality
round(accuracy(linear.trend.seasonality.for$fitted, retailsales.ts), 3)

# Accuracy for quadratic trend and Seasonality
round(accuracy(quadratic.trend.seasonality.for$fitted, retailsales.ts), 3)

# Accuracy performance measure for the model = "ZZZ"
round(accuracy(HW.ZZZ.pred$fitted, retailsales.ts), 3)

# two level
round(accuracy(quadratic.trend.seasonality$fitted + residual.ar1$fitted, retailsales.ts), 3)

# Arima 
round(accuracy(arima.seas.pred$fitted, retailsales.ts), 3)

# Auto Arima
round(accuracy(auto.arima.full.pred$fitted, retailsales.ts), 3)

# Seasonal naive forecast
round(accuracy((snaive(retailsales.ts))$fitted, retailsales.ts), 3)

# Naive forecast.
round(accuracy((naive(retailsales.ts))$fitted, retailsales.ts), 3)

