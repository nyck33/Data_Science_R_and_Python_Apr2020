NUM_SIMS=1000
SAMPLE_SIZE=5
LAMBDA = 10
poissonAverages=numeric(NUM_SIMS)
for (i in 1:NUM_SIMS){
  sample=rpois(n=SAMPLE_SIZE,
               lambda=LAMBDA)
  poissonAverages[i]=mean(sample)
}
#par(mfrow=c(2,1))
#hist(sample)
#hist(poissonAverages,xlim=range(sample))
#boxplot(sample,poissonAverages)
var(poissonAverages)


######################## code on made up distribution
NUM_SIMS=1000
SAMPLE_SIZE=200
wackypopulation=c(
  runif(100),
  runif(100)+2,
  runif(100)+5)


waitingAverages=numeric(NUM_SIMS)
for (i in 1:NUM_SIMS){
  sampleOfWaitingTimes=
    sample(x=wackypopulation,
           size=SAMPLE_SIZE,
           replace=TRUE)
  waitingAverages[i]=mean(sampleOfWaitingTimes)
}
par(mfrow=c(2,1))
hist(sampleOfWaitingTimes,breaks=10)
hist(waitingAverages,xlim=range(wackypopulation))
summary(waitingAverages)

