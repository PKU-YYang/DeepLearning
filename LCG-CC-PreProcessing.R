

zeroone_normalizaiton=function(x){
  x=scale(x,scale=F)
  #y=whiten(features)$U
  y=(x-min(x))/(max(x)-min(x))
  
  return(y)
}




train_preprocessing=function(pl_threshold,num_clusters,train_test_ratio,
                       training_data,label_no){
  
  pl_risk_threshold=pl_threshold # 200 1-2%
  noofclu=num_clusters
  train_ratio=train_test_ratio
  
  cat("Reading in the Training Data:", training_data,'\n\n')
  
  clients=read.csv(training_data,header=T) 

  #永恒不变的输入的列数
  n_col=ncol(clients)
  
  clients[clients[,label_no]>=pl_risk_threshold,ncol(clients)+1]=paste('H')
  clients[clients[,label_no]<pl_risk_threshold,ncol(clients)]=paste('L')
  colnames(clients)[ncol(clients)]='PLRiskDefination'



  rratio=dim(clients[clients$PLRiskDefination=='H',])[1]/nrow(clients)

  cat('Defination of High Risky Clients: P&L >',pl_risk_threshold,
      ', they occupy',rratio*100,'% of the total','\n\n')

  Sys.sleep(5)
  cat('Number of Sub-classifiers:',noofclu,'\n\n')

  Sys.sleep(5)
  cat(train_ratio*100,'% of the data used to train, ', (1-train_ratio)*100,
      '% of the data to validate','\n\n')


  dummy=c()
  
  #calculate the dummy variables in the features
  #features=1:19
  for (i in 1:(label_no-1)){
    
    if (length(unique(clients[,i]))<10){
      
      dummy=c(dummy,i)
    }
    
  }
  

  #dummy=18:19
  ncol=ncol(clients)
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

#colnames(clients)

  clientsplit=split(clients,clients$PLRiskDefination)
# 1[[H]]  2 [[L]]


  clu=scale(clientsplit[[2]][,label_no:n_col])

#clusters_features_to_use 21:25

  cat('Clustering low risky clients into',noofclu,
      'clusters based on these',length(label_no:n_col),'features: \n\n',
      colnames(clients)[label_no:n_col],'\n\n')

  fit=kmeans(clu,noofclu,iter.max=100)

  clientsplit[[2]][,ncol(clients)+1]=fit$cluster
  colnames(clientsplit[[2]])[ncol(clients)+1]="ClientCluster"

  cluster=split(clientsplit[[2]],
                clientsplit[[2]]$ClientCluster)
#####################show the result of clustering############
sink('cluster_log.txt')

cat("P&L threshold:",pl_risk_threshold,'\n\n')

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
    label=c(cluster[[i]][,ncol(cluster[[i]])],clabel) # 
    #归一化不一样的label，0是high risky，其他的是各种Low risky
    label=ifelse(label==0,0,1)
    
    
    if (length(dummy)!=0){
    multiple_dataset_index=c(setdiff(1:(label_no-1),dummy),
                             (n_col+2):(ncol(cluster[[1]])-1))}else{
                               multiple_dataset_index=c(1:(label_no-1))  
                             }
    #ncol+1:避开riskdefiniation
    #ncol(clients):clientcluster 23456
  
    
    multiple_dataset[[i]]=rbind(cluster[[i]][,unique(multiple_dataset_index)],
                            clientsplit[[1]][,unique(multiple_dataset_index)])
    
  
    pl_test_set[[i]]=c(cluster[[i]][,label_no+1],
                         clientsplit[[1]][,label_no+1]) 
    multiple_dataset[[i]]=sapply(multiple_dataset[[i]],zeroone_normalizaiton)
    #label是粘在后面的
    multiple_dataset[[i]]=cbind(multiple_dataset[[i]],label)
  }


for (i in 1:noofclu){
  filename=paste('../dpdata/multiple_dl_',i,sep="")
  trainno=sample(nrow(multiple_dataset[[i]]),
                 train_ratio*nrow(multiple_dataset[[i]]))
  
  testno=setdiff(1:nrow(multiple_dataset[[i]]),trainno)
  train_set=multiple_dataset[[i]][trainno,]
  valid_set=multiple_dataset[[i]][testno,]
  test_set=pl_test_set[[i]][testno]
  
  trainname=paste(filename,"_train.csv",sep='')
  validname=paste(filename,"_valid.csv",sep='')
  testname=paste(filename,"_test.csv",sep='') #没有label
  testplname=paste(filename,"_test_PL.csv",sep='') #有label有Pl
  #set test_set as both valid and test set
  write.csv(train_set,file=trainname,row.names=F)
  write.csv(valid_set,file=validname,row.names=F)
  write.csv(valid_set[,1:(ncol(valid_set)-1)],file=testname,row.names=F)
  write.csv(test_set,file=testplname,row.names=F)
  
  cat('In sub classifier',i,':',100*dim(train_set[train_set[,ncol(train_set)]==0,])[1]
      /nrow(train_set),'% are high risky clients\n\n' )
}
sink()

cat('Details of clusters have been saved in cluster_log.txt','\n\n')

return(ncol(train_set)-1)}





#这个是extemd dataset

extend_preprocessing=function(training_data,extending_data,non_feature){

cat("Reading in the Extending Data:", extending_data,'\n\n')

extend_clients=read.csv(extending_data,header=T)

extend_clients=extend_clients[,1:(non_feature-1)]
#cat("Reading in the Training Data:", training_data,'\n\n')

clients=read.csv(training_data,header=T) 


dummy=c()

#calculate the dummy variables in the features
#features=1:19
for (i in 1:ncol(extend_clients)){
  
  if (length(unique(extend_clients[,i]))<10){
    
    dummy=c(dummy,i)
  }  
}


ncol=dim(extend_clients)[2]
n_col=ncol
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

if (length(dummy)!=0){
extend_dataset_index=c(setdiff(1:n_col,dummy),
                       (n_col+1):ncol(extend_clients))}else {
                         
                         extend_dataset_index=c(1:n_col)
                         
                       }



extend_set=sapply(extend_clients[,unique(extend_dataset_index)],
                zeroone_normalizaiton)

write.csv(extend_set,
          file='../dpdata/multiple_extend_set.csv',
          row.names=F)

return(ncol(extend_set))

}





