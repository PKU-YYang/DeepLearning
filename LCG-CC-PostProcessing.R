
#要注意拼接的是哪个PL条件下的结果

#setwd("~/PycharmProjects/DeepLearningTutorials/dpdata")

PL_cost=function(par=0.5,data){
  
  #decision=c()
  
  
  data=cbind(data,ifelse(data[,2]>par,0,1)) # 1表示对赌
  
  colnames(data)[3]="Decision"
  
  cost=sum(data[,3]*data[,1])
  #     
  #     for (i in 1:(ncol(data)-1)){
  #       
  #       decision=cbind(decision,ifelse(data[,i+1]>par[i],0,1))
  #       # 概率是等于0的概率
  #     } 
  #   
  #     decision_all=apply(decision,1,sum) # 0-high risk clients
  #     
  #     cost_all=tapply(data[,1],decision_all,sum)
  #     
  #    cost=-sum(data[,1])+cost_all[1]
  return(cost)  # optim minimize,the smaller the cost, bigger lcg can earn
  
  
}

postprocessing=function(num_clusters,extending_data,non_feature){

noofclu=num_clusters

filehead="multiple_dl_"

plfile="_test_PL.csv"

resultfile="_result_on_valid.csv"

best_threshold=c()

#calculating different threshold for sub-classifiers

cat('\n','Adjusting different threshold for sub-classifiers','\n')

for (i in 1:noofclu){
  origin_result=read.csv(paste(filehead,i,plfile,sep=''),header=T)
  predict_result=read.csv(paste(filehead,i,resultfile,sep=''),header=F)
  predict_result[,3]=ifelse(predict_result[,1]==0,
                            predict_result[,2],1-predict_result[,2])
  #只保留等于零的概率，也就是high risky clients的概率
  to_optimize=cbind(origin_result,predict_result[,3])
  colnames(to_optimize)=c("PL20","Prob_to_be_high")
  
  result=optim(par=0.5,PL_cost,data=to_optimize,
               method="Brent",lower=0,upper=1)
  
  
  best_threshold=c(best_threshold,result[[1]])

  cat("Classfier",i ,": After adjustation, with threshold =",result[[1]],
      ", P&L is increased from", 
      -sum(to_optimize[,1]),"to",
      -result[[2]],"\n")
}


fileextend="multiple_dl_"

raw_extendset=read.csv(extending_data,header=T)
final_result=cbind(raw_extendset[,non_feature:ncol(raw_extendset)])


for (i in 1:noofclu){
    
  extend_result=read.csv(paste(fileextend,i,"_extend.csv",sep=""),
                           header=F)
  
  #calculate the prob being high risky
  extend_result[,3]=ifelse(extend_result[,1]==0,
                           extend_result[,2],1-extend_result[,2])
  
  #adjust according to the threshold
  extend_result[,4]=ifelse(extend_result[,3]>best_threshold[i],0,1)
  
  final_result=cbind(final_result,extend_result[,c(1,3,4)])
  
  colnames(final_result)[(ncol(final_result)-2):ncol(final_result)]=c(
    paste("Classifier_",i,sep=""),"Prob_being_high_risky",
    paste("Adjusted_classifier_",i,"_",sprintf("%.2f",best_threshold[i]),sep=""))
  
}

no_nona=seq(4,4+3*(noofclu-1),3)
no_a=seq(6,6+4*(noofclu-1),3)

Scores_Adjusted=noofclu
Scores_NonAdjusted=noofclu
for (i in 1:noofclu){
  Scores_Adjusted=Scores_Adjusted-final_result[,no_a[i]]
  
  Scores_NonAdjusted=Scores_NonAdjusted-final_result[,no_nona[i]]
}

to_write=cbind(Scores_Adjusted=Scores_Adjusted,Scores_NonAdjusted=Scores_NonAdjusted,final_result)

write.csv(to_write,"dp_final_result.csv",row.names=F)
}



