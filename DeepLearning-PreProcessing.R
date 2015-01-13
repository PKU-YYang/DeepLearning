
zeroone_normalizaiton=function(x){
  x=scale(x,scale=F)
  #y=whiten(features)$U
  y=(x-min(x))/(max(x)-min(x))
  
  return(y)
}

extend_preprocessing=function(training_data,extending_data,non_feature){
  
  
  #cat("Reading in the Training Data:", training_data,'\n\n')
  
  clients=read.csv(training_data,header=T) 
  
  cat("Reading in the Extending Data:", extending_data,'\n\n')
  
  extend_clients=read.csv(extending_data,header=T)
  
  extend_clients=extend_clients[,1:(non_feature-1)]
 
  #create the dummy variable in the extending data
  ncol=dim(extend_clients)[2]
  n_col_extend=dim(extend_clients)[2]
  
  dummy=c()
  
  for (i in 1:n_col_extend){
    
    if (length(unique(extend_clients[,i]))<10){
      
      dummy=c(dummy,i)
    }
    
  } 
  
  
  
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
  #n_col_extend比non_feature小一
  extend_dataset_index=c(setdiff(1:(n_col_extend),dummy),
                         (n_col_extend+1):ncol(extend_clients))
  
  #没有label所以可以直接归一化，多Label不可以直接归一化
  extend_set=sapply(extend_clients[,unique(extend_dataset_index)],
                    zeroone_normalizaiton)
  } else {
    extend_set=sapply(extend_clients,zeroone_normalizaiton)
  
  } 
  
  write.csv(extend_set,file='../dpdata/DP_extend.csv',row.names=F)
  
  return(c(ncol(extend_set),length(unique(clients[,non_feature]))))
}






train_preprocessing=function(train_test_ratio,training_data,label_no){
  
  cat("Reading in the Training Data:", training_data,'\n\n')
  
  clients=read.csv(training_data,header=T) 
  
  clients=clients[,1:label_no]
  
  dummy=c()

  n_col_train=ncol(clients)
  ncol=ncol(clients)
#calculate the dummy variables in the features

for (i in 1:(label_no-1)){
  
  if (length(unique(clients[,i]))<10){
    
    dummy=c(dummy,i)
  }
  
} 

#create the dummy variable in the training data

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


if (length(dummy)!=0){ 
  #n_col_train和label_no一样大
  train_dataset_index=c(setdiff(1:(label_no-1),dummy),
                         (label_no+1):ncol(clients))
  train_dataset=sapply(clients[,unique(train_dataset_index)],
                    zeroone_normalizaiton)
  
} else {
  train_dataset=sapply(clients[,1:(label_no-1)],zeroone_normalizaiton)
} 

#label不可以一起送进去归一化
train_dataset=cbind(train_dataset,label=clients[,label_no])

trainno=sample(nrow(train_dataset),train_test_ratio*nrow(train_dataset))
testno=setdiff(1:nrow(train_dataset),trainno)

train_set=train_dataset[trainno,]
test_set=train_dataset[testno,]

write.csv(train_set,"../dpdata/DP_train.csv",row.names=F)
write.csv(test_set,"../dpdata/DP_valid.csv",row.names=F)

return(c(ncol(train_set)-1,length(unique(clients[,label_no]))))


}
