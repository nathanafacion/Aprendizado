if (!require("twitteR")) {
  install.packages("twitteR", repos="http://cran.rstudio.com/") 
  library("twitteR")
}

consumer_key <- "XsL85oXFjlhpjGoog1jgAyZ1o"
consumer_secret <- "QeYCrJhA0bfwFqSo3OPyzHHHzlClPtBawFFKKhnwrHPPJ9frDB"
access_token <- "61682678-22WqzGgczKQHMVKJFbhqGa2n4xZ1KiDwDAkl6mHNf"
access_secret <- "GAsx23xIB4G2e0vHInoZAPHBlxuSORGyLHlG2LYRTVlgZ"
options(httr_oauth_cache=T) #This will enable the use of a local file to cache OAuth access credentials between R sessions.
setup_twitter_oauth(consumer_key,
                    consumer_secret,
                    access_token,
                    access_secret)

nanafacion <- getUser("nanafacion")
location(nanafacion)

nanafacion_followers<-nanafacion$getFollowers(retryOnRateLimit=300)

nanafacion_followers

length(nanafacion_followers)

nanafacion = rbindlist(lapply(lucaspuente_follower_IDs,as.data.frame))

if (!require("data.table")) {
  install.packages("data.table", repos="http://cran.rstudio.com/") 
  library("data.table")
}

nanafacion_followers_location = rbindlist(lapply(nanafacion_followers,as.data.frame))

head(nanafacion_followers_location$location, 133)

nanafacion_followers_location$location<-gsub("%", " ",nanafacion_followers_location$location)

#Install key package helpers:
source("https://raw.githubusercontent.com/LucasPuente/geocoding/master/geocode_helpers.R")
#Install modified version of the geocode function
#(that now includes the api_key parameter):
source("https://raw.githubusercontent.com/LucasPuente/geocoding/master/modified_geocode.R")

geocode_apply<-function(x){
  geocode(x, source = "google", output = "all", api_key="[INSERT YOUR GOOGLE API KEY HERE]")
}
