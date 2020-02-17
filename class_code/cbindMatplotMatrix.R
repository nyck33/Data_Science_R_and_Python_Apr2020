jeter=cbind(c(12,48),c(183,582),c(190,654))
justice = cbind(c(104,411),c(45,140),c(163,495))

typeof(jeter) #double
class(jeter) #matrix

jeterproportions = jeter[1,]/jeter[2,]
justiceproportions=justice[1,]/justice[2,]

jeterproportions
justiceproportions

class(jeterproportions) #numeric
typeof(justiceproportions) #double

combined = cbind(jeterproportions,justiceproportions)
matplot(combined,type="l")

jeteroverall = rowSums(jeter)
justiceoverall = rowSums(justice)
class(jeteroverall)
typeof(jeteroverall)

jeteroverall[1]/jeteroverall[2]
justiceoverall[1]/justiceoverall[2]


### Code using array ####
alldata = array(c(justice,jeter),dim=c(2,3,2))#dim=hits,year,person (rows, cols, ch)
alldata
combined = alldata[1,,]/alldata[2,,] #divide row 1 by row 2
combined
matplot(combined,type="l")

# apply() same function to several vars in df
# sum alldata over rows and channels
overall = apply(alldata,c(1,3),sum)
# sum alldata over rows so row 1 in ch1 and 2, row2 in ch 1 and 2 for 2 sums
row_sum = apply(alldata,c(1),sum)
row_sum
# 3 cols so 3 sums sum of each col over 2 ch's
col_sum = apply(alldata,c(2),sum)
col_sum

overall
overall[1,]/overall[2,]
