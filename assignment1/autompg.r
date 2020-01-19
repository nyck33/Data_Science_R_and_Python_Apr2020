autompg = read.csv(file="/home/nobu/Desktop/stat3990/assignment1/autompg.ssv", sep=" ")
# keep only rows with no missing values
completeRows = complete.cases(autompg)
autompg = autompg[completeRows, ]
dim(autompg)
# make histogram of mpg 
hist(autompg$mpg, breaks= 15, xlab="mpg", ylab="num models", main="Distribution of Fuel Efficiencies")

# convert to L/100km and make a boxplot
lPer100km <-235/autompg$mpg
# bind to data frame for easy access
autompg = cbind(lPer100km, autompg)
# make box plot, label and title it 
boxplot(autompg$lPer100km, xlab="num models", ylab="Litres per 100km", main="Fuel Efficiency")
#Num cylinders and mpg side by side
unique(autompg$cyl)
eightCyl<-subset(autompg, cyl==8)
sixCyl<-subset(autompg,cyl==6)
fiveCyl<-subset(autompg,cyl==5)
fourCyl<-subset(autompg,cyl==4)
threeCyl<-subset(autompg,cyl==3)
par(mfrow=c(2,3))
hist(eightCyl$mpg, breaks=8, xlab="mpg",ylab="count", main="mpg: 8 cylinder engines")
hist(sixCyl$mpg, breaks=5, xlab="mpg",ylab="count", main="mpg: 6 cylinder engines")
hist(fiveCyl$mpg, breaks=4, xlab="mpg",ylab="count", main="mpg: 5 cylinder engines")
hist(fourCyl$mpg, breaks=7, xlab="mpg",ylab="count", main="mpg: 4 cylinder engines")
hist(threeCyl$mpg,breaks=3, xlab="mpg",ylab="count", main="mpg: rotary engines (Mazda)")

#num cylinders vs Lper100km
par(mfrow=c(2,3))
hist(eightCyl$lPer100km, breaks=8, xlab="Lper100km",ylab="count", main="Lper100km: 8 cyl engines")
hist(sixCyl$lPer100km, breaks=5, xlab="Lper100km",ylab="count", main="Lper100km: 6 cyl engines")
hist(fiveCyl$lPer100km, breaks=4, xlab="Lper100km",ylab="count", main="Lper100km: 5 cyl engines")
hist(fourCyl$lPer100km, breaks=7, xlab="Lper100km",ylab="count", main="Lper100km: 4 cyl engines")
hist(threeCyl$lPer100km, breaks=3, xlab="Lper100km",ylab="count", main="Lper100km: rotary engines")

#parallel coordinate plot in blue-orange scale for Lper100km
blueOrange=blueOrange <-colorRampPalette(c("orange", "blue"))
theColours = blueOrange(length(lPer100km))
coloursToPaint = theColours[rank(lPer100km)]
