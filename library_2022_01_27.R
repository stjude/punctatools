
#########################################################
# Define a function for the analysis of this data set

puncta.stats=function(dset,                   # data.frame of experimental data
                     y.var,                  # name of y variable in dset
                     cond.var,               # name of condition variable in dset
                     tech.rep,               # name of subsample variable
                     cond.inc,               # vector of conditions to include in analysis
                     trans.name="identity",  # name of function to transform y as a string
                     y.count=F)              # indicates (T/F) whether y is a count variable
                    
{
  # define the variables
  dset$xvar1=dset[,cond.var]
  dset$xvar2=dset[,tech.rep]
  dset$yvar = dset[,y.var]
  
 
  # subset
  keep=(dset$xvar1%in%cond.inc)&
       (!is.na(dset$xvar1))&
       (!is.na(dset$xvar2))&
       (!is.na(dset$yvar))
  temp=dset[keep,]
  
  n.clr = length(unique(temp$xvar1)) # number of colors
  
  # calculate median over tech replicates by condition
  
  med = aggregate(yvar~xvar1+xvar2, data=temp, median)
  grp = med[,1]
  y.med = med[,3]
  
  
  if (!y.count)                                
  {
     res=lmer(yvar~xvar1+(1|xvar2),       
              data=temp)
     rpt=report.lmer.result(res,
                            y.var,
                            cond.var,
                            trans.name)  
     
     par(mfrow=c(1,2))
   
     boxplot(y.med~grp, col=rainbow(n.clr), 
             ylab="Subsample medians",
             xlab = "", main=y.var)
     text(n.clr/2, max(y.med)/1.15,paste0("p value= ", 
                                          round(rpt$anova.table$`Pr(>F)`,6)), cex=0.75)
     
     boxplot(yvar ~ xvar1 + xvar2, data = temp,
             col = rainbow(n.clr),xaxt = "n",
             main=y.var, xlab="Condition by technical replicates")
     text(length(unique(temp$xvar2))/2, 
          max(temp$yvar)/1.15,
          paste0("p value= ", 
                 round(rpt$anova.table$`Pr(>F)`,6)), cex=0.75)
     
     return(rpt)                                      
  }
  
  if (y.count)                                 
  {
     res=glmer(yvar~xvar1+(1|xvar2),       
               data=temp,
               family=poisson())
     null.res = glmer(yvar~(1|xvar2),       
                      data=temp,
                      family=poisson())
     rpt=report.glmer.result(res,null.res,
                             y.var,cond.var)
     
     par(mfrow=c(1,2))
   
     boxplot(y.med~grp, col=rainbow(n.clr), 
             ylab="Subsample medians",
             xlab = "", main=y.var)
     text(n.clr/2, max(y.med)/1.15,paste0("p value= ", 
                                          round(rpt$anova.table$`Pr(>Chisq)`[2],6)), cex=0.75)
     
     boxplot(yvar ~ xvar1 + xvar2, data = temp,
             col = rainbow(n.clr),xaxt = "n",
             main=y.var, xlab="Condition by technical replicates")
     text(length(unique(temp$xvar2))/2, 
          max(temp$yvar)/1.15,
          paste0("p value= ", 
                 round(rpt$anova.table$`Pr(>Chisq)`[2],6)), cex=0.75)
     return(rpt)
  }
}

############################################
# Generate tabular results and suggested narrative statements for lmer model results

report.lmer.result=function(lmer.result,
                            y.var,
                            cond.var,
                            trans.name)
{
  res=lmer.result                                  # define res variable to simplify code below
  anova.tbl=anova(res)                             # ANOVA result
  
  emm = emmeans(res, ~ xvar1)
  mc = data.frame(summary(pairs(emm)))
  mc$lowerCI = mc$estimate - 1.96*mc$SE
  mc$upperCI = mc$estimate + 1.96*mc$SE
  comb.tbl = mc
  emm.tbl = data.frame(emm)
  
  re.tbl = ranef(res)                              # random effect
  
  # produce the usual narrative interpretation
  y.name=y.var
  if (trans.name!="identity") 
          y.name=paste0(trans.name,"(",
                        y.name,")")
  
  aov.sig=c("significantly differed",
            "did not significantly differ")[1+(anova.tbl["Pr(>F)"]>0.05)]
  
  cond.names=as.character(unique(model.frame(res)[,"xvar1"]))
  cond.names1 = sub(' -.*', '', comb.tbl$contrast)
  cond.names2 = sub('.*- ', '', comb.tbl$contrast)
  
  if (length(cond.names)==2) 
     cond.str=paste0(cond.names,collapse=" and ")
  if (length(cond.names)>2)
  {
     cond.str=paste0(cond.names[-length(cond.names)],collapse=", ")
     cond.str=paste0(cond.str,", and ",cond.names[length(cond.names)])
  }
  
  aov.statement=paste0("The mean ",y.name," ",aov.sig,
                       " across the conditions ",
                       cond.str," (p = ",anova.tbl["Pr(>F)"],").")
   
  tbl.statement=paste0("The mean ",y.name," of the condition ",
                       cond.names1," was ",
                       c(""," not ")[1+(comb.tbl[,"p.value"]>0.05)],
                       "significantly ",
                       c("less than ","greater than ")[1+(comb.tbl[,"estimate"]>0)],
                       "that of the condition ",cond.names2," (diff = ",comb.tbl[,"estimate"],
                       "; p = ",comb.tbl[,"p.value"],"; 95% CI: ",
                       comb.tbl[,"lowerCI"],", ",comb.tbl[,"upperCI"],").")
  
  res=list(narrative=c(aov.statement,
                       tbl.statement),
           LSmean.table = emm.tbl,
           estimate.table=comb.tbl,
           anova.table=anova.tbl,
           random.table=re.tbl)
  return(res)
  
}

###################################
# report glmer result

report.glmer.result=function(glmer.result,
                             null.result,
                             y.var, cond.var)
{
  res=glmer.result
  anova.tbl=NA
  anova.tbl=anova(res, null.result)                      # ANOVA result
  
  emm = emmeans(res, ~ xvar1)
  mc = data.frame(summary(pairs(emm)))
  mc$lowerCI = mc$estimate - 1.96*mc$SE
  mc$upperCI = mc$estimate + 1.96*mc$SE
  comb.tbl = mc
  emm.tbl = data.frame(emm)
  
  re.tbl = ranef(res)
  
  # produce the usual narrative interpretation
  #y.name=paste0("log10(",y.var,")")
  y.name=y.var
  aov.sig=c("significantly differed",
            "did not significantly differ")[1+(anova.tbl[-1,"Pr(>Chisq)"]>0.05)]
  
  cond.names=as.character(unique(model.frame(res)[,"xvar1"]))
  if (length(cond.names)==2) 
          cond.str=paste0(cond.names,collapse=" and ")
  if (length(cond.names)>2)
  {
          cond.str=paste0(cond.names[-length(cond.names)],collapse=", ")
          cond.str=paste0(cond.str,", and ",cond.names[length(cond.names)])
  }
  
  aov.statement=paste0("The mean ",y.name," ",aov.sig,
                       " across the conditions ",
                       cond.str," (p = ",anova.tbl[-1,"Pr(>Chisq)"],").")
 
  tbl.statement=paste0("The mean ",y.name," of the condition ",
                       cond.names1," was ",
                       c(""," not ")[1+(comb.tbl[,"p.value"]>0.05)],
                       "significantly ",
                       c("less than ","greater than ")[1+(comb.tbl[,"estimate"]>0)],
                       "that of the condition ",cond.names2," (diff = ",comb.tbl[,"estimate"],
                       "; p = ",comb.tbl[,"p.value"],"; 95% CI: ",
                       comb.tbl[,"lowerCI"],", ",comb.tbl[,"upperCI"],").")
  
  res=list(narrative=c(aov.statement,
                       tbl.statement),
           LSmean.table = emm.tbl,
           estimate.table=comb.tbl,
           anova.table=anova.tbl,
           random.table=re.tbl)
  
  return(res)
  
}