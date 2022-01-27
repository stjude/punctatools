#############################################################
### Author:     William Wu
### Date:       01-21-2022
### Purpose:    Run Puncta function
### PI:         Richard Kriwachi
#############################################################

rm(list=ls())


#################################################################
#####   data  prep ###

library(readxl)
root = "//biosshr-srvr/GenomicProjects2/"
data.dir = paste0(root,"WilliamWu/RichardKriwaski/Rpackage/")
HEK.file = "puncta_features_NHA9_HEK293T_ST_011722.xlsx"
HEK.df = read_xlsx(paste0(data.dir, HEK.file),sheet = 1,col_names = TRUE)
HEK.df = data.frame(HEK.df)

HEK.sub = HEK.df[,c("condition","sample",
                    "Mutual.information.Hoechst.vs.GFP",
                    "FL.light.phase.conc.GFP",
                    "FL.puncta.number.per.nuclear.vol.GFP")]
HEK.sub$tech.rep = sub('.*Position', '',HEK.sub$sample)
HEK.sub$tech.rep = gsub(".*[_]([^.]+)[.].*", "\\1", HEK.sub$tech.rep)
HEK.sub$tech.rep = as.factor(as.numeric(HEK.sub$tech.rep))

# data transformation 
HEK.sub$FL.light.phase.conc.GFP = log(HEK.sub$FL.light.phase.conc.GFP+0.01)
HEK.sub$FL.puncta.number.per.nuclear.vol.GFP = round(HEK.sub$FL.puncta.number.per.nuclear.vol.GFP,0)

# puncta features for analysis
y.vars=c("Mutual.information.Hoechst.vs.GFP",
         "FL.light.phase.conc.GFP",
         "FL.puncta.number.per.nuclear.vol.GFP"
         )

# # specify transformations of those y-variables
y.trans=c("identity","log", "identity")

# specify whether the y-variables are count variables
y.cnt = c(F,F,T)

# show the analysis specifications matrix
analysis.specs=cbind.data.frame(y.var=y.vars,
                                y.trans=y.trans,
                                y.cnt=y.cnt)

analysis.specs


#################### Run puncta ###############
library(tidyverse)
library(lme4)
library(lmerTest)

options("scipen"=100, "digits"=4)

cond.inc=c("NHX9","NHX9_21FGAA","NHX9_8FA")

source("Q:/WilliamWu/RichardKriwaski/Rpackage/library_2022_01_27.R")
i=2
fit0=try(puncta.stats(HEK.sub,
                      y.var=y.vars[i],
                      trans.name=y.trans[i],
                      cond.inc=cond.inc,
                      cond.var="condition",
                      tech.rep="sample",
                      y.count=y.cnt[i]))

# reference 
# https://pubmed.ncbi.nlm.nih.gov/34903620/

library(emmeans)

f=lmer(yvar~xvar1+(1|xvar2),  data=temp)
emm = emmeans(f, ~ xvar1)
mc = data.frame(summary(pairs(emm)))
mc$lowerCI = mc$estimate - 1.96*mc$SE
mc$upperCI = mc$estimate + 1.96*mc$SE

temp1 = temp[temp$condition !="NHX9",]
f1 = lmer(yvar~xvar1+(1|xvar2),  data=temp1)
emm1 = emmeans(f1, ~ xvar1)
mc1 = data.frame(summary(pairs(emm1)))


g=glmer(yvar~xvar1+(1|xvar2),       
          data=temp,
          family=poisson())
emm2 = emmeans(g, ~xvar1)
mc2 = data.frame(summary(pairs(emm2)))

