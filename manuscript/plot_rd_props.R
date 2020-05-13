rm(list=ls())

library(data.table)
library(stringr)
library(ggplot2)

classifiers <- list("bernoulli_nb"="Bernoulli Naive Bayes")
classifiers <- list("rf"="Random Forest")
DATE <- "2020_05_07"

prep_data <- function(df, group_cols, int_cause_col, prop_type){
  df <- fread(df)
  if (paste0(int_cause, "_deaths_thesis") %in% names(df)){
  setnames(df, paste0(int_cause, "_deaths_thesis"), int_cause)
  }
  df <- df[, lapply(.SD,sum), 
           by=group_cols, 
           .SDcols=int_cause]
  df[, prop := get(int_cause)/sum(get(int_cause))]
  df[, prop_type := paste0(prop_type)]
  return(df)
}

plot_data <- function(int_cause, short_name){
  rd <- prep_data(df=paste0("/home/j/temp/agesak/thesis/model_results/", int_cause, "_gbd_2019.csv"),
                  group_cols="cause_name", int_cause_col = int_cause, prop_type="GBD 2019")
  df <- prep_data(df=paste0("/home/j/temp/agesak/thesis/model_results/", DATE, "/", DATE, "_", int_cause, "_", short_name, "_predictions.csv"),
                  group_cols="cause_name", int_cause_col = paste0(int_cause, "_deaths_thesis"), prop_type=classifiers[[short_name]])
  all_df <- rbind(rd, df)
  colors <- c("GBD 2019"="paleturquoise1", classifier_name="rosybrown1")
  names(colors)[c(2)] <- c(classifiers[[short_name]])
  all_df$prop_type <- factor(all_df$prop_type, levels = c("GBD 2019", classifiers[[short_name]]))
  
  plot <- ggplot(all_df, aes(x=reorder(cause_name, -prop), y=prop, fill=prop_type)) +
    geom_bar(position=position_dodge(), stat="identity", colour='black') + 
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(), axis.line = element_line(colour = "black"),
          axis.text.x = element_text(angle = 45, hjust = 1, size=10)) +
    scale_x_discrete(labels = function(x) str_wrap(x, width = 17)) + xlab("Cause Name") +
      scale_fill_manual("Model Type", values=colors) + 
    ylab(paste(toupper(int_cause), "proportion")) + ggtitle(paste(toupper(int_cause), "Redistribution Fractions")) + 
    labs(fill="Model Type") + scale_y_continuous(expand = c(0,0))
  
  dir.create(paste0("/home/j/temp/agesak/thesis/figures/", DATE, "/"), showWarnings = FALSE)
  ggsave(paste0("/home/j/temp/agesak/thesis/figures/", DATE, "/", DATE, "_", int_cause, "_", short_name, "_compare_results.pdf"), plot, dpi=300, height=12, width=19)
  
}
for (int_cause in c("x59", "y34")){
  for (short_name in names(classifiers)){
      print(paste("working on", int_cause, short_name))
      plot_data(int_cause, short_name) 
  }
}
