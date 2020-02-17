# jan 27 code
#rnorm (n, mu, var)
toronto=rnorm(365,1,10)
victoria=rnorm(365,1,5)
hist(toronto)
x11()
hist(victoria,xlim=range(victoria))

sd(toronto)
sd(victoria)
var(toronto)
var(victoria)
### moving to iris data
plot(iris$Petal.Width,iris$Petal.length,col=as.integer(iris$Species))
# what is pc????
pc = prcomp(iris$Petal.Width)
pc
plot(pc)

pc_toronto = prcomp(toronto)
pc_toronto
plot(pc_toronto)
