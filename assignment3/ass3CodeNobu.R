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

#not normally distributed, left skewed
hist(versicolor$sepal.wid, breaks=10)

sample = versicolor[sample(nrow(versicolor), 30),]

#student's t-test assumes same variance
t.test(sample$sepal.wid, versicolor$sepal.wid, var.equal = TRUE)

# Welch's t-test assumes different variance
t.test(versicolor1$sepal.wid, versicolor2$sepal.wid, var.equal = FALSE)


# take random samples from versicolor and run t tes
# Question 3, 2 sided t test 

virginica <- iris[iris$class =='Iris-virginica',]

remove.packages("ggpubr")

#Welch 2 sample t-test
t.test(versicolor$sepal.wid, virginica$sepal.wid, alternative = "two.sided", var.equal = FALSE)

#question 4 permutation test
total <- rbind(versicolor, virginica)

library(coin)
set.seed(101)
permutation_test <- independence_test(sepal.wid ~ class, data=total)
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

#question 8
#replicate 1000 times
pvalues <- vector()
for(i in 1:10000){
  expdist<-rexp(10, rate=1)
  pvalues[i] = t.test(expdist, mu=1)$p.value
}

expdist
head(pvalues)
pvalues
hist(pvalues)

small_pvalues <- pvalues[pvalues < 0.1]
small_pvalues
hist(small_pvalues)

#question 10
num_samps = 10000
points <- matrix(0, nrow=num_samps, ncol=2)
#increment n
n=0

for(i in 1:num_samps){
  temp <- (1-(1/n))**n
  points[i,1] = n
  points[i,2] = temp
  n = n+1
}
points
plot(points, xlim=c(0,100), ylim=c(0.2, 0.4), main='bootstrap P(not pick datum)', xlab='n', ylab='P')

