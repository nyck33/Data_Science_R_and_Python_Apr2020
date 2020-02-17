# read in csv and change col names
iris = read.csv(file="/home/nobu/Desktop/stat3990/data/iris.data")
plot(iris[,3:4])

#install.packages("plyr")
library("plyr")
iris<-rename(iris, c("X5.1"="sepal.len", "X3.5"="sepal.wid", "X1.4"="petal.len", "X0.2"="petal.wid", "Iris.setosa"="class"))
save(iris, file="iris.rda")
load("iris.rda")

#divide Versicolor data into 2 groups
versicolor <- iris[iris$class =='Iris-versicolor',]
versicolor

#not normally distributed, left skewed
hist(versicolor$sepal.wid, breaks=10)

sample = versicolor[sample(nrow(versicolor), 30),]
sample

#student's t-test assumes same variance
t.test(sample$sepal.wid, versicolor$sepal.wid, var.equal = TRUE)
# Two Sample t-test
# 
# data:  sample$sepal.wid and versicolor$sepal.wid
# t = 0.65927, df = 58, p-value = 0.5123
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   -0.1425386  0.2825386
# sample estimates:
#   mean of x mean of y 
# 2.84      2.77 

# Welch's t-test assumes different variance
t.test(versicolor1$sepal.wid, versicolor2$sepal.wid, var.equal = FALSE)


# take random samples from versicolor and run t tes
# Question 3, 2 sided t test 

virginica <- iris[iris$class =='Iris-virginica',]

remove.packages("ggpubr")

#Welch 2 sample t-test
t.test(versicolor$sepal.wid, virginica$sepal.wid, alternative = "two.sided", var.equal = FALSE)
# Welch Two Sample t-test
# 
# data:  versicolor$sepal.wid and virginica$sepal.wid
# t = -3.2058, df = 97.927, p-value = 0.001819
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   -0.33028364 -0.07771636
# sample estimates:
#   mean of x mean of y 
# 2.770     2.974 

#question 4 permutation test
total <- rbind(versicolor, virginica)

library(coin)
set.seed(101)
permutation_test <- independence_test(sepal.wid ~ class, data=total)
permutation_test
#hist(permutation_test, breaks=10)

# Asymptotic General Independence Test
# 
# data:  sepal.wid by
# class (Iris-versicolor, Iris-virginica)
# Z = -3.0654, p-value = 0.002174
# alternative hypothesis: two.sided

# try to plot the permutation
#https://mac-theobio.github.io/QMEE/permutation_examples.html
nsim <- 1000
res <- numeric(nsim)

for(i in 1:nsim){
  perm <- sample(nrow(total))
  bdat <- transform(total, sepal.wid = sepal.wid[perm])
  
  res[i] <- mean(bdat[bdat$class=="Iris-versicolor","sepal.wid"]) - 
    (mean(bdat[bdat$class=="Iris-virginica", "sepal.wid"]))
}

obs <- mean(total[total$class=="Iris-versicolor", "sepal.wid"]) -
  mean(total[total$class=="Iris-virginica", "sepal.wid"])

res <- c(res,obs)

hist(res,col="gray", las=1, main="")
abline(v=obs,col="red")

#question 5

# use aov
res_petallen <- aov(petal.len ~ class, data=total)
summary(res_petallen)
#use oneway.test
oneway.test(petal.len ~class, data = total)
#student's t-test assumes same variance
t.test(petal.len ~ class, data=total, var.equal = TRUE)

#petal width
# use aov
res_petalwid <- aov(petal.wid ~ class, data=total)
summary(res_petalwid)
#use oneway.test
oneway.test(petal.wid ~class, data = total)
#student's t-test assumes same variance
t.test(petal.wid ~ class, data=total, var.equal = TRUE)

# use aov
res_sepallen <- aov(sepal.len ~ class, data=total)
summary(res_sepallen)
#use oneway.test
oneway.test(sepal.len ~class, data = total)
#student's t-test assumes same variance
t.test(sepal.len ~ class, data=total, var.equal = TRUE)

# use aov
res_sepalwid <- aov(sepal.wid ~ class, data=total)
summary(res_sepalwid)
#use oneway.test
oneway.test(sepal.wid ~class, data = total)
#student's t-test assumes same variance
t.test(sepal.wid ~ class, data=total, var.equal = TRUE)




t.test()
#question 6
x = rep(0,10); x[1]=1
x
#Welch's
t.test(x, alternative = "two.sided", var.equal = FALSE)
#Student's t-test
t.test(x, var.equal = TRUE)

#question 7
library(boot)
# function to obtain mean of x
fc <- function(x, indices){
  dt <- x[indices]
  c(mean(dt))
}
set.seed(123)
mybootstrap <- boot(x, fc, R=1000)
head(mybootstrap)
plot(mybootstrap$t, index=1, main="Boostrap mean of x", xlab="iteration", ylab="bootstrap mean")
hist(mybootstrap$t, breaks = 10, main="Histogram of Bootstrap mean of X", xlab="mean", xlim = c(-0.1, 0.6))
summary(mybootstrap)
boot.ci(mybootstrap, index=1, type='basic')
# BOOTSTRAP CONFIDENCE INTERVAL CALCULATIONS
# Based on 1000 bootstrap replicates
# 
# CALL : 
#   boot.ci(boot.out = mybootstrap, index = 1)
# 
# Intervals : 
#   Level      Normal              Basic         
# 95%   (-0.0922,  0.2854 )   (-0.1000,  0.2000 )  
# 
# Level     Percentile            BCa          
# 95%   ( 0.0,  0.3 )   ( 0.0,  0.3 )  
# Calculations and Intervals on Original Scale

#question 8
#Estimating type I Error

expdist <- rexp(10, rate=1)
plot(expdist)
head(expdist)
t.test(expdist, mu=1)$p.value

#replicate 1000 times
pvalues <- vector()
for(i in 1:10000){
  expdist<-rexp(10, rate=1)
  pvalues[i] = t.test(expdist, mu=1)$p.value
}

head(pvalues)
pvalues
hist(pvalues)

small_pvalues <- pvalues[pvalues < 0.1]
small_pvalues
hist(small_pvalues)
