library(FactoMineR)
library(taRifx)
library(ForeCA)
library(outliers)
library(bigmemory)   


 setwd("~/PycharmProjects/SQL")
# clients=read.csv('./multiple_deeplearning.csv',header=T)
# 
# extend_clients=read.csv('./multiple_deeplearning_extend.csv'
#                        ,header=T)

load("~/PycharmProjects/SQL/multiple_dp.RData")


#PL defination threshold
pl_risk_threshold=1100 # 200 1-2%
noofclu=6
train_ratio=0.9


zeroone_normalizaiton=function(x){
  x=scale(x,scale=F)
  #y=whiten(features)$U
  y=(x-min(x))/(max(x)-min(x))
  
  return(y)
}





clients[clients$NextTotalPL_GBP>=pl_risk_threshold,ncol(clients)+1]=paste('H')
clients[clients$NextTotalPL_GBP<pl_risk_threshold,ncol(clients)]=paste('L')
colnames(clients)[ncol(clients)]='PLRiskDefination'
#head(clients)

rratio=dim(clients[clients$PLRiskDefination=='H',])[1]/nrow(clients)
cat("the risk ratio is: ",rratio)

ncol=ncol(clients)
dummy=18:19
for (i in dummy){
  
  for (j in unique(clients[,i])){
    #cat(colnames(clients)[i],j,'\n')
    coln=paste(colnames(clients)[i],j,sep='')
    clients[,ncol+1]=0
    clients[clients[,i]==j,ncol+1]=1
    ncol=ncol+1
    colnames(clients)[ncol]=coln
  }
}

colnames(clients)

clientsplit=split(clients,clients$PLRiskDefination)
# 1[[H]]  2 [[L]]


clu=scale(clientsplit[[2]][,21:25])
#summary(clientsplit[[2]][,22:25])


fit=kmeans(clu,noofclu)

#####################show the result of clustering############
clientsplit[[2]][,ncol(clients)+1]=fit$cluster
colnames(clientsplit[[2]])[ncol(clients)+1]="ClientCluster"


clugbp=tapply(clientsplit[[2]]$NextTotalPL_GBP,
       clientsplit[[2]]$ClientCluster,mean)

cluplcon=tapply(clientsplit[[2]]$NextTotalPL_GBP,
               clientsplit[[2]]$ClientCluster,sum)

cluwr=tapply(clientsplit[[2]]$WinningRate,
       clientsplit[[2]]$ClientCluster,mean)


clusharp=tapply(clientsplit[[2]]$SharpNext100,
       clientsplit[[2]]$ClientCluster,mean)


clutrade=tapply(clientsplit[[2]]$TotalTrades,
       clientsplit[[2]]$ClientCluster,mean)

cluster=split(clientsplit[[2]],
              clientsplit[[2]]$ClientCluster)

cluno=lapply(cluster,nrow)


for (i in 1:noofclu) {
  
  cat('Cluster',i,":",'\n',"Number",
      as.numeric(cluno[i])/nrow(clients)*100,'%\n',
      "P&L",clugbp[i],'\n',"WinningRate",
      cluwr[i],'\n', 
      "SharpRatio",clusharp[i],'\n',"TotalTrades"
      ,clutrade[i],'\n','P&L Contribution'
      ,cluplcon[i]/sum(cluplcon)*100,'%'
      ,'\n\n')
}
##############################################################



#把cluster的数据和hish risky client粘起来
#这里 high risky client 是 0
clabel=ifelse(clientsplit[[1]]$PLRiskDefination=='H',0,1)
multiple_dataset=list()
pl_test_set=list()
for (i in 1:noofclu){
  #整合两组不同的label
  label=c(cluster[[i]][,39],clabel) # 39 is clientcluster
  label=ifelse(label==0,0,1)
  #归一化不一样的label，0是high risky，其他的是各种Low risky
  multiple_dataset[[i]]=rbind(cluster[[i]][,c(1:17,29:38)],
                            clientsplit[[1]][,c(1:17,29:38)])
  
  pl_test_set[[i]]=rbind(cluster[[i]][,c(26,27,21)],
                         clientsplit[[1]][,c(26,27,21)]) #第21列是nextplgbp_20
  multiple_dataset[[i]]=sapply(multiple_dataset[[i]],zeroone_normalizaiton)
  multiple_dataset[[i]]=cbind(multiple_dataset[[i]],label)
  
}


for (i in 1:noofclu){
  filename=paste('../DeepLearningTutorials/dpdata/multiple_dl_',i,sep="")
  trainno=sample(nrow(multiple_dataset[[i]]),
                 train_ratio*nrow(multiple_dataset[[i]]))
  
  testno=setdiff(1:nrow(multiple_dataset[[i]]),trainno)
  train_set=multiple_dataset[[i]][trainno,]
  valid_set=multiple_dataset[[i]][testno,]
#   test_set=multiple_dataset[[i]][testno,] 
#   test_set=cbind(test_set,pl_test_set[[i]][testno])#test多一列pl
#   colnames(test_set)[29]='PLNEXT20'
  test_set=pl_test_set[[i]][testno,]
  
  trainname=paste(filename,"_train.csv",sep='')
  validname=paste(filename,"_valid.csv",sep='')
  testname=paste(filename,"_test.csv",sep='') #没有label
  testplname=paste(filename,"_test_PL.csv",sep='') #有label有Pl
  #set test_set as both valid and test set
  write.csv(train_set,file=trainname,row.names=F)
  write.csv(valid_set,file=validname,row.names=F)
  write.csv(valid_set[,1:(ncol(valid_set)-1)],file=testname,row.names=F)
  write.csv(test_set,file=testplname,row.names=F)
  
  cat('Multiple',i,':',100*dim(train_set[train_set[,ncol(train_set)]==0,])[1]
      /nrow(train_set),'%\n' )
}




#这个是extemd dataset
dummy=18:19
ncol=dim(extend_clients)[2]
for (i in dummy){
  
  for (j in unique(clients[,i])){ #如果clients和extend_clients的segment 
                                  #marketcluster不一样，这里的i要改
    #cat(colnames(clients)[i],j,'\n')
    coln=paste(colnames(extend_clients)[i],j,sep='')
    extend_clients[,ncol+1]=0
    extend_clients[extend_clients[,i]==j,ncol+1]=1
    ncol=ncol+1
    colnames(extend_clients)[ncol]=coln
  }
}




extend_set=sapply(extend_clients[,c(1:17,23:32)],
                zeroone_normalizaiton)

write.csv(extend_set,
          file='../DeepLearningTutorials/dpdata/multiple_extend_set.csv',
          row.names=F)

colnames(extend_set)
colnames(train_set)
colnames(valid_set)
colnames(test_set)




