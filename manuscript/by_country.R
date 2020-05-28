# redistribution fractions by country
rm(list=ls())

library(data.table)
library(ggplot2)
library(RColorBrewer)
library(stringr)
library(wesanderson, lib.loc="/homes/agesak/R/3.5/")

# CHANGE THE COLORS AND PUT IN A LOOP

DATE <- "2020_05_23_most_detailed"
classifiers <- list("bernoulli_nb"="Bernoulli Naive Bayes", "nn"="Neural Network")
# classifiers <- list("nn"="Neural Network")

prep_data <- function(df, int_cause, short_name){
  
  # can either show top 5 causes overall
  by_cause <- df[, lapply(.SD,sum), 
                 by="cause_name", 
                 .SDcols=int_cause]
  by_cause[, prop := get(int_cause)/sum(get(int_cause))]
  # what a gross mix of data table and data frame
  # top 5 causes over all
  top_causes <- tail(by_cause[order(prop)], 5)$cause_name
  # top_causes <- tail(by_cause[order(prop)], 6)$cause_name
  # drop because of south africa
  # top_causes <- top_causes[!top_causes == "Adverse effects of medical treatment"]
  df <- df[cause_name %in% top_causes]
  
  # groupby country/cause
  df2 <- df[, lapply(.SD,sum), 
            by=c("location_name", "cause_name"), 
            .SDcols=int_cause]
  # get prop by country
  df2[, prop :=  get(int_cause)/sum(get(int_cause)), by = "location_name"]
  # # zero motorcycle injuries in ZAF - dropped ZAF
  # if (int_cause == "y34"){
  #   dt <- data.table(location_name="South Africa", prop=0, cause_name="Motorcyclist road injuries", y34=0)
  #   # ONLY DO THIS WHEN DROPPING ADVERSE EFFECTS
  #   # dt2 <- data.table(location_name="South Africa", prop=0, cause_name="Pedestrian road injuries", y34=0)
  #   df2 <- rbind(df2, dt)
  #   # df2 <- rbind(df2, dt, dt2)
  # }
  # order props by country
  df2 <- df2[order(cause_name, prop)]
  
  # so the bars will be ordered within groups
  df2[, ID := rep(c(1:6), 5)]
  
  # OR can get top 5 causes by country - kinda messy though
  # dt3 <- data.table(df2, key=c("location_name", "prop"))
  # dt3 <- dt3[, tail(.SD, 5), by=c("location_name")]
  return(df2)
}

plot_data <- function(int_cause, short_name){
  df <- fread(paste0("/home/j/temp/agesak/thesis/model_results/", DATE, "/", DATE, "_", int_cause, "_", short_name, "_predictions.csv"))
  setnames(df, paste0(int_cause, "_deaths_thesis"), int_cause)
  
  df <- prep_data(df, int_cause, short_name)
  # colors <- c('#D83151', '#26A146', '#FCAA51', '#1CC2BD', '#D354AC', '#9CC943', '#006966')
  # darksalmon, "plum2",  "skyblue", , "yellow2"
  colors <- c("lightpink", "sandybrown", "thistle2","navajowhite", "palegreen2", "mediumaquamarine")
  
  
  plot <- ggplot(df, aes(x=reorder(cause_name, -prop), y=prop, fill=location_name, group=ID)) + 
    geom_bar(position=position_dodge(), stat="identity", colour='black') +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(), axis.line = element_line(colour = "black"),
          axis.text.x = element_text(angle = 45, hjust = 1, size=16, color="black"),
          axis.text.y = element_text(size=16, color="black"),
          axis.title=element_text(size=20), legend.text=element_text(size=20),
          legend.title=element_text(size=20), plot.title = element_text(size=22)) + scale_x_discrete(labels = function(x) str_wrap(x, width = 25)) + 
    xlab("Cause Name") + labs(fill="Location Name") + ylab(paste(toupper(int_cause), "proportion")) + 
    scale_fill_manual(values = colors) + scale_y_continuous(expand = c(0, 0)) + ggtitle("Top 5 Causes")

  # wes_palette("Royal2", 7, type = c("continuous"))
  # scale_fill_brewer(palette = "Paired")
  dir.create(paste0("/home/j/temp/agesak/thesis/figures/", DATE, "/"), showWarnings = FALSE)
  ggsave(paste0("/home/j/temp/agesak/thesis/figures/", DATE, "/", DATE, "_", int_cause, "_", short_name, "_by_country.pdf"), plot, dpi=300, height=12, width=22)
  
}

for (int_cause in c("x59")){
  for (short_name in names(classifiers)){
    print(paste("working on", int_cause, short_name))
    plot_data(int_cause, short_name) 
  }
}

# df <- fread(paste0("/home/j/temp/agesak/thesis/model_results/", DATE, "/", DATE, "_", int_cause, "_", short_name, "_predictions.csv"))
# setnames(df, paste0(int_cause, "_deaths_thesis"), int_cause)
# 
# 
# # can either show top 5 causes overall
# by_cause <- df[, lapply(.SD,sum), 
#           by="cause_name", 
#           .SDcols=int_cause]
# by_cause[, prop := get(int_cause)/sum(get(int_cause))]
# # what a gross mix of data table and data frame
# # top 5 causes over all
# top_causes <- tail(by_cause[order(prop)], 5)$cause_name
# df <- df[cause_name %in% top_causes]
# 
# # groupby country/cause
# df2 <- df[, lapply(.SD,sum), 
#          by=c("location_name", "cause_name"), 
#          .SDcols=int_cause]
# # get prop by country
# df2[, prop :=  get(int_cause)/sum(get(int_cause)), by = "location_name"]
# # zero motorcycle injuries in ZAF
# if (int_cause == "y34"){
#   dt <- data.table(location_name="South Africa", prop=0, cause_name="Motorcyclist road injuries", y34=0)
#   df2 <- rbind(df2, dt)
# }
# # order props by country
# df2 <- df2[order(cause_name, prop)]
# 
# # so the bars will be ordered within groups
# df2[, ID := rep(c(1:7), 5)]
# 
# 
# # OR can get top 5 causes by country - kinda messy though
# # dt3 <- data.table(df2, key=c("location_name", "prop"))
# # dt3 <- dt3[, tail(.SD, 5), by=c("location_name")]
# 
# plot <- ggplot(df2, aes(x=reorder(cause_name, -prop), y=prop, fill=location_name, group=ID)) + 
#   geom_bar(position=position_dodge(), stat="identity", colour='black') +
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"),
#         axis.text.x = element_text(angle = 45, hjust = 1, size=16, color="black"),
#         axis.text.y = element_text(size=16, color="black"),
#         axis.title=element_text(size=20), legend.text=element_text(size=20),
#         legend.title=element_text(size=20), plot.title = element_text(size=22)) + scale_x_discrete(labels = function(x) str_wrap(x, width = 25)) + 
#   xlab("Cause Name") + labs(fill="Location Name") + ylab(paste(toupper(int_cause), "proportion")) + 
#   scale_fill_brewer(palette = "BuPu") + scale_y_continuous(expand = c(0, 0))
# 
# dir.create(paste0("/home/j/temp/agesak/thesis/figures/", DATE, "/"), showWarnings = FALSE)
# ggsave(paste0("/home/j/temp/agesak/thesis/figures/", DATE, "/", DATE, "_", int_cause, "_", short_name, "_by_country.pdf"), plot, dpi=300, height=12, width=22)
# 
