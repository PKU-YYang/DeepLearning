setwd("~/PycharmProjects/DeepLearningTutorials/dpdata")

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
  return(cost)  # optim minimize
  
  
}



noofclu=6

filehead="multiple_dl_"

plfile="_test_PL.csv"

resultfile="_result_on_valid.csv"

best_threshold=c()

for (i in 1:noofclu){
  origin_result=read.csv(paste(filehead,i,plfile,sep=''),header=T)
  predict_result=read.csv(paste(filehead,i,resultfile,sep=''),header=F)
  predict_result[,3]=ifelse(predict_result[,1]==0,
                            predict_result[,2],1-predict_result[,2])
  #只保留等于零的概率，也就是high risky clients的概率
  to_optimize=cbind(origin_result$NextTotalPL_GBP20,predict_result[,3])
  colnames(to_optimize)=c("PL20","Prob_to_be_high")
  
  result=optim(par=0.5,PL_cost,data=to_optimize,
               method="Brent",lower=0,upper=1)
  
  
  best_threshold=c(best_threshold,result[[1]])

  cat("Classfier",i ,"with par =",result[[1]],", P&L from", 
      -sum(to_optimize[,1]),"to",
      -result[[2]],"\n")
}


fileextend="multiple_dl_"

raw_extendset=read.csv("multiple_deeplearning_extend.csv",header=T)
final_result=cbind(raw_extendset$accountid,raw_extendset$Period,
               as.numeric(raw_extendset$NextTotalPL_GBP20))
colnames(final_result)=c('Accountid','Period',"PL_Next20")

for (i in 1:noofclu){
    
  extend_result=read.csv(paste(fileextend,i,"_extend.csv",sep=""),
                           header=F)
    
  extend_result[,3]=ifelse(extend_result[,1]==0,
                           extend_result[,2],1-extend_result[,2])
  
  extend_result[,4]=ifelse(extend_result[,3]>best_threshold[i],0,1)
  
  final_result=cbind(final_result,extend_result[,c(1,3,4)])
  
  colnames(final_result)[(ncol(final_result)-2):ncol(final_result)]=c(
    paste("Classifier_",i,sep=""),paste("Prob_",i,sep=""),
    paste("Decision_",i,"_",sprintf("%.2f",best_threshold[i]),sep=""))
  
}

write.csv(final_result,"mutiple_final_result.csv",row.names=F)



