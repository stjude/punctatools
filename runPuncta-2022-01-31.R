#############################################################
### Author:     William Wu
### Date:       01-21-2022
### Purpose:    Run Puncta function
### PI:         Richard Kriwachi
#############################################################

rm(list=ls())


#################################################################
#####   Read in data  from GitHub site directly       ###

library(readr)
urlfile="https://raw.githubusercontent.com/stjude/punctatools/main/example_data/thermodynamic_characterization/puncta_thermo_features.csv"
HEK.df <-read_csv(url(urlfile))
HEK.df = data.frame(HEK.df)

HEK.sub = HEK.df[,c("condition",
                    "sample",
                    "Mutual.information.Hoechst.vs.GFP",
                    "light.phase.conc.GFP",
                    "puncta.number.per.nuclear.vol.GFP")]

##  data transformation 
HEK.sub$light.phase.conc.GFP = log(HEK.sub$light.phase.conc.GFP)
HEK.sub$puncta.number.per.nuclear.vol.GFP = round(HEK.sub$puncta.number.per.nuclear.vol.GFP,0)

# puncta features for analysis
y.vars=c("Mutual.information.Hoechst.vs.GFP",
         "light.phase.conc.GFP",
         "puncta.number.per.nuclear.vol.GFP"
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


######################################################################
#################### Run puncta ######################################
library(tidyverse)
library(lme4)
library(lmerTest)
library(emmeans)

options("scipen"=100, "digits"=4)

# source the R script

library(devtools)
devtools::source_url("https://github.com/stjude/punctatools/blob/main/library_2022_01_31.R?raw=TRUE")


### Run the function ###

cond.inc=c("NHX9","NHX9_21FGAA","NHX9_8FA")

i=3
fit0=try(puncta.stats(HEK.sub,
                      y.var=y.vars[i],
                      trans.name=y.trans[i],
                      cond.inc=cond.inc,
                      cond.var="condition",
                      tech.rep="sample",
                      y.count=y.cnt[i]))


