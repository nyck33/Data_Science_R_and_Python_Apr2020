mortality = read.csv(file.choose())
mortality

plot(mortality, type="l", main="Child mortality over time", xlab="year", ylab="mortality rate")
#question 2, p = died at age 2, 

# probs of dying at 2 and 70
p_die_2 <-mortality[1,2]
p_die_2

p_die_70 <- (1.0 - p_die_2)
p_die_70

life_expectancy <- (p_die_2 * 2) + (p_die_70 * 70)
life_expectancy

# make table with array??? no

# prob_table <-array(0, dim=c(2,3))
# prob_table[1,1] = 'X'
# prob_table

# by matrix
prob_table <-matrix(c(2, 70,p_die_2, p_die_70), ncol=2, byrow=TRUE)
rownames(prob_table)<-c("X", "p(X)")
prob_table
# save matrix to file
write.table(prob_table, file="prob_table.txt", sep=" ", row.names = TRUE, col.names = TRUE)

# avg life expectancy in 1800
prob_table[1,1] # chyeck values
prob_table[1,2]
life_expectancy_1800 <- prob_table[1,1] * prob_table[2,1]  + prob_table[1,2] *prob_table[2,2] 
life_expectancy_1800

# add column for life expectancy using the formula for each year and justify what happens question 3
die_at_2 <- mortality[,2] * 2.0
die_at_70 <- (1.0 - mortality[,2]) * 70.0
life_expectancy <- die_at_2 + die_at_70
life_expectancy

# add column
mortality$expect_life = life_expectancy
plot(mortality[,c(1,3)], type='l', main="Life Expectancy Over Time", xlab='year', ylab='life expectancy')
# questions 4 no.  Once people made it passed the vulnerable age of 2 to 5, they lived fairly long on avg

# Smoking
# read table: https://www.datacamp.com/community/tutorials/r-data-import-tutorial#txt
sfev <- read.table("http://jse.amstat.org/datasets/fev.dat.txt", 
                          col.names = c ('age', "fev", "ht", "sex", "smoke"))
sfev

summary(sfev)
head(sfev)
nrow(sfev)
#subset data
# density for smokers and non-smokers FEV
smokers <-subset(sfev, smoke==1)
smokers_dens <- density(smokers$fev)
#https://stackoverflow.com/questions/45845660/superimposing-two-or-more-subsets-in-same-plot
nonsmokers<- subset(sfev, smoke==0)
nonsmokers_dens <-density(nonsmokers$fev)

# mean FEV and age for smokers
smokers_mean_FEV = mean(smokers$fev)
smokers_mean_FEV
smokers_mean_Age = mean(smokers$age)
smokers_mean_Age
smokers_mean_Height = mean(smokers$ht)
smokers_mean_Height
num_smokers = nrow(subset(sfev, smoke==1))
num_smokers
# female smokers
num_f_smokers = nrow(subset(sfev, smoke==1 & sex==0))
num_f_smokers
num_m_smokers = nrow(subset(sfev, smoke==1 & sex==1))
num_m_smokers

# mean FEV and age for nonsmokers
nonsmokers_mean_FEV = mean(nonsmokers$fEV)
nonsmokers_mean_FEV
nonsmokers_mean_Age = mean(nonsmokers$age)
nonsmokers_mean_Age
nonsmokers_mean_Ht = mean(nonsmokers$ht)
nonsmokers_mean_Ht
num_nonsmokers = nrow(subset(sfev, smoke==0))
num_nonsmokers

# Kinda uselss overall density
plot(smokers_dens, main="Distribution Density of FEV", xlab="FEV", col="red")
lines(nonsmokers_dens, col="blue")
legend("topright", legend=c("Smokers", "Nonsmokers"), col=c("red", "blue"), lty=c(1,1))


#boxplot
par(mfrow=c(1,2), mar)
boxplot(sfev$fev[sfev$smoke==1])

#densities by age###############################################################
plot(sfev[,1:4])
plot(sfev$fev[sfev$age>=12 | sfev$age<=20], col=sfev$smoke)
plot((smokers$fev[smokers$age>=12 & smokers$age<=20]),col=("red"))
plot(nonsmokers$fev[nonsmokers$age>=12 & nonsmokers$age<=20], col="blue")

#densities by height#####################################################
smoker_hbins <- smokers[,"ht"]
cut(smoker_hbins, 4, include.lowest = TRUE, labels=c("short","med","med+","tall"))

#58 to 61.5 inches
plot(density(smokers$fev[smokers$ht>=58 & smokers$ht<=61.5]), col='red')
lines(density(nonsmokers$fev[nonsmokers$ht>=58 & nonsmokers$ht<=61.5]), col='blue')

#61.6 to 65 inches
plot(density(smokers$fev[smokers$ht>=61.6& smokers$ht<=65.0]), col='red')
lines(density(nonsmokers$fev[nonsmokers$ht>=61.6 & nonsmokers$ht<=65.0]), col='blue')
#65.1 inches to 68.5
plot(density(smokers$fev[smokers$ht>=65.1 & smokers$ht<=68.5]), col='red')
lines(density(nonsmokers$fev[nonsmokers$ht>=65.1 & nonsmokers$ht<=68.5]), col='blue')

#68.6 to 72
plot(density(smokers$fev[smokers$ht>=68.6 & smokers$ht<=72.0]), col='red')
lines(density(nonsmokers$fev[nonsmokers$ht>=68.6 & nonsmokers$ht<=72.0]), col='blue')

#58 to 72
plot(density(smokers$fev[smokers$ht>=58 & smokers$sex==1]), col='red', main="FEV by Height",xlab="FEV",ylab='Density', ylim=c(0, 0.7), xlim=c(0, 7))
lines(density(nonsmokers$fev[nonsmokers$ht>=58 & smokers$sex==0]), col='blue')
legend("topright", legend=c("smoker m", "nonsmoker f"), col=c("red", "blue"),lty=c(1,1))

# by gender and height
plot(density(sfev$ht[sfev$sex==1]), col='red', ylim=c(0,0.1), main='Height by Sex', xlab="Height", ylab="Density")
lines(density(sfev$ht[sfev$sex==0]), col='blue')
legend("topright", legend=c("male", "female"), col=c("red", "blue"),lty=c(1,1))


# densities by sex and ht > 58 inches
#male is red, female blue
#par(mar=c(5,4,4,2)+0.1)
plot(density(nonsmokers$fev[nonsmokers$sex==1 & smokers$ht > 58]), col='red', main="Nonsmokers FEV by Sex", xlab='FEV', ylab='Density',ylim=c(0,0.6))
lines(density(nonsmokers$fev[nonsmokers$sex==0 & smokers$ht > 58]), col='blue')
legend("topright", legend=c("male", "female"), col=c("red", "blue"),lty=c(1,1))

plot(density(smokers$fev[smokers$sex==1 & smokers$ht > 58]), col='red', main='Smokers FEV by Sex', xlab='FEV', ylab='Density', ylim=c(0,1.2))
lines(density(smokers$fev[smokers$sex==0 & smokers$ht > 58]), col='blue')
legend("topright", legend=c("male", "female"), col=c("red", "blue"),lty=c(1,1))

#males over 58 inches smokers vs nonsmokers
plot(density(smokers$fev[smokers$sex==1 & smokers$ht > 58]), col='red', main='Male Smokers and Nonsmokers', xlab='FEV', ylab='Density', xlim=c(0, 6.5))
lines(density(nonsmokers$fev[nonsmokers$sex==1 & nonsmokers$ht > 58]), col ='blue')

# femaies same as above
plot(density(smokers$fev[smokers$sex==0 & smokers$ht > 58]), col='red', main='Female Smokers and Nonsmokers', xlab='FEV', ylab='Density', xlim=c(0, 6.5))
lines(density(nonsmokers$fev[nonsmokers$sex==0 & nonsmokers$ht > 58]), col ='blue')

#males heights who smoke vs nonsmokers
plot(density(sfev$ht[sfev$smoke==1]), col='red', xlim=c(40,80), main="Height Smokers Nonsmokers", xlab='Height', ylab='Density')
lines(density(sfev$ht[sfev$smoke==0]), col='blue')

# no idea what this is
plot(x=sfev$ht, y=sfev$fev, col=sfev$smoke)
legend(legend=unique(sfev$smoke))
hist(sfev$fev, breaks=20, col=sfev$smoke)
# height vs FEV is positive
plot(x=sfev$ht, y=sfev$fev, xlab="Height", ylab="FEV")
#linear regression shows high residuals in normal distribution
HtFEVresult <- lm(smoking_FEV$Ht.Inches ~ smoking_FEV$FEV)
HtFEVresult
hist(HtFEVresult$residuals, breaks=20)
# age vs FEV is positive
plot(x=smoking_FEV$Age, y=smoking_FEV$FEV,  xlab="Age", ylab="FEV")
# regr shows residuals to be long right skewed
AgeFEVresult <- lm(smoking_FEV$Age ~ smoking_FEV$FEV)
AgeFEVresult
hist(AgeFEVresult$residuals, breaks=20)
###################################################################################
# parallel coordinate plot doesn't really help
library(MASS)
parcoord(smoking_FEV[,], col=smoking_FEV$Smoke, var.label = TRUE)
####################################################################################



