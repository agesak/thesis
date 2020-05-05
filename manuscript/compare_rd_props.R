rm(list=ls())

library(data.table)
library(stringr)
library(ggplot2)

classifiers <- list("multi_nb"="Multinomial Naive Bayes")


prep_data <- function(df, group_cols, prop_type){
  df <- fread(df)
  df <- df[, lapply(.SD,sum), 
           by=group_cols, 
           .SDcols=int_cause]
  df[, prop := get(int_cause)/sum(get(int_cause))]
  df[, prop_type := paste0(prop_type)]
  return(df)
}

for (int_cause in c("y34", "x59")){
  for (short_name in c("multi_nb")){
    print(paste("working on", int_cause, short_name))
    df <- prep_data(df=paste0("/home/j/temp/agesak/thesis/", int_cause, "_", short_name, "_rd.csv"),
                             group_cols="cause_name", prop_type=classifiers[[short_name]])
    rd <- prep_data(df=paste0("/home/j/temp/agesak/thesis/", int_cause, "_", short_name, "_predictions.csv"),
                    group_cols="cause_name", prop_type="GBD 2019")
    all_df <- rbind(rd, df)
    plot <- ggplot(all_df, aes(x=reorder(cause_name, -prop), y=prop, fill=prop_type)) + 
      geom_bar(position=position_dodge(), stat="identity", colour='black') + 
      theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            panel.background = element_blank(), axis.line = element_line(colour = "black"),
            axis.text.x = element_text(angle = 45, hjust = 1, size=10)) +
      scale_x_discrete(labels = function(x) str_wrap(x, width = 17))
    
    ggsave(paste0("/home/j/temp/agesak/thesis/figures/", short_name, "_compare_results_", int_cause , ".pdf"), plot, dpi=300, height=12, width=17)
    
  }
  
}
# rd <- fread(paste0("/home/j/temp/agesak/thesis/", int_cause, "_", short_name, "_rd.csv"))
# df <- 
# 
# group_cols = c("cause_name")
# 
# df <- df[, lapply(.SD,sum), 
#     by=group_cols, 
#     .SDcols=int_cause]
# df[, prop := get(int_cause)/sum(get(int_cause))]
# rd <- rd[, lapply(.SD,sum), 
#          by=group_cols, 
#          .SDcols=int_cause]
# rd[, prop := get(int_cause)/sum(get(int_cause))]
# 
# 
# rd[, prop_type := "GBD 2019"]
# df[, prop_type := short_name]
# 
# all_df <- rbind(rd, df)
# 
# plot <- ggplot(all_df, aes(x=reorder(cause_name, -prop), y=prop, fill=prop_type)) + 
#          geom_bar(position=position_dodge(), stat="identity", colour='black') + 
#   theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
#         panel.background = element_blank(), axis.line = element_line(colour = "black"),
#         axis.text.x = element_text(angle = 45, hjust = 1, size=10)) +
#   scale_x_discrete(labels = function(x) str_wrap(x, width = 17))
# 
# ggsave(paste0("/home/j/temp/agesak/thesis/figures/", short_name, "_compare_results_", int_cause , ".pdf"), plot, dpi=300, height=12, width=17)
# 
# 
