library(fpp3)
library(regclass)
library(dplyr)
library(ggplot2)

### DATA LOADING AND MANIPULATION ###

ashbase <- read.csv("asheville-airbnb-reviews.csv")
denbase <- read.csv("denver-airbnb-reviews.csv")


## Ash Data manipulation
ashbase$date <- make_yearmonth(year = ashbase$year, month = ashbase$month)

ash <- NA

ash <- ashbase %>% group_by(date) %>% summarise(count = n())

colnames(ash) <- c("date","asheville_number_of_reviews")

## Denver Data Manipulation
denbase$date <- make_yearmonth(year = denbase$year, month = denbase$month)

den <- NA

den <- denbase %>% group_by(date) %>% summarise(count = n())

colnames(den) <- c("date","denver_number_of_reviews")

## Joining data

fulldata <- full_join(ash,den, by = "date")

fulldata$total_number_of_reviews <- fulldata$asheville_number_of_reviews + fulldata$denver_number_of_reviews

## Ended up dropping these to make it easier predicting. Possibly remove from Code.
fulldata$asheville_number_of_reviews <- NULL
fulldata$denver_number_of_reviews <- NULL

fulldata <- fulldata %>% 
  as_tsibble()




### PREDICTIONS ###

#Splitting data

holdout_size <- 12

train <- head(fulldata, -holdout_size)
holdout <- tail(fulldata, holdout_size)

# # Validating split
# max(train$date)
# min(holdout$date)
# 
# nrow(train)
# nrow(holdout)



### Smoothing for covid outliers

decomp <- fulldata %>%
  model(STL(total_number_of_reviews, robust = TRUE)) %>%
  components(total_number_of_reviews)

# decomp %>% autoplot
# 
# decomp %>% autoplot(trend + season_year)

fulldata$total_number_of_reviews <- (decomp$trend + decomp$season_year)




## SELECTING BEST MODEL

#Fitting several models
lambda <- fulldata %>% 
  features(total_number_of_reviews, features = guerrero) %>% 
  pull(lambda_guerrero)

train_cv <- train %>% 
  stretch_tsibble(.init = 48 , .step = 12)

fits <- train_cv %>% 
  model(
    TSLM_ts = TSLM(box_cox(total_number_of_reviews, lambda) ~ trend() + season()),
    tslm_tf = TSLM(box_cox(total_number_of_reviews, lambda) ~ trend() + fourier(K = 6)),
    arima = ARIMA(box_cox(total_number_of_reviews, lambda)),
    ets = ETS(box_cox(total_number_of_reviews, lambda)),
    arima = ARIMA(box_cox(total_number_of_reviews, lambda)),
    nnetar = NNETAR(box_cox(total_number_of_reviews, lambda))
  )


fits %>%
  forecast(h = 12, times = 0) %>%
  accuracy(train) %>%
  arrange(RMSE)


#Fitting best model
ets_fit <- fulldata %>% 
  model(
    ets = ETS(box_cox(total_number_of_reviews, lambda))
  )

# #Checking out residuals/white noise
# report(ets_fit)
# 
# gg_tsresiduals(ets_fit)

# Plot with full data
ets_fit %>% 
  forecast(h = 12) %>% 
  autoplot(fulldata)




### Storing Predictions to CSV

CSV <- data.frame(1:12)

CSV$Date <- c("Jan 2021", "Feb 2021", "March 2021", "April 2021", "May 2021",
              "June 2021", "July 2021", "August 2021", "September 2021", 
              "October 2021", "November 2021", "December 2021")


predictions <- ets_fit %>% 
  forecast(h = 12)

CSV$Predictions <- predictions$.mean

CSV$X1.12 <- NULL

write.csv(CSV,file='Final Group Predictions.csv',row.names=FALSE)