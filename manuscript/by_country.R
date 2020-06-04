# redistribution fractions by country for top 5 causes (figure 5)
rm(list=ls())

library(data.table)
library(ggplot2)
library(RColorBrewer)
library(stringr)

DATE <- "2020_05_23_most_detailed"
CLASSIFIERS <- list("xgb"="Gradient Boosting", "multi_nb"="Naive Bayes", "rf"="Random Forest", "nn"="Deep Neural Network")

prep_data <- function(df, int_cause, short_name){
  
  # group by cause name
  by_cause <- df[, lapply(.SD,sum), 
                 by="cause_name", 
                 .SDcols=int_cause]
  by_cause[, prop := get(int_cause)/sum(get(int_cause))]

  # get total deaths by country
  df[, total_deaths := sum(get(int_cause)), by="location_name"]
  
  # top 5 causes over all
  top_causes <- tail(by_cause[order(prop)], 5)$cause_name
  df2 <- df[cause_name %in% top_causes]
  
  # groupby country/cause
  df2 <- df2[, lapply(.SD,sum), 
            by=c("location_name", "cause_name"), 
            .SDcols=int_cause]
  
  # merge on total deaths
  df2 <- merge(df2, unique(df[, c("location_name", "total_deaths")]), by="location_name")
 
  # get prop by country
  df2[, prop := get(int_cause)/total_deaths, by = "location_name"]
  
  # order props by country
  df2 <- df2[order(cause_name, prop)]
  
  # so the bars will be ordered within groups
  df2[, ID := rep(c(1:6), 5)]
  
  return(df2)
}

plot_data <- function(int_cause, short_name){
  df <- fread(paste0("/home/j/temp/agesak/thesis/model_results/", DATE, "/", DATE, "_", int_cause, "_", short_name, "_predictions.csv"))
  setnames(df, paste0(int_cause, "_deaths_thesis"), int_cause)
  
  df <- prep_data(df, int_cause, short_name)
  colors <- c("lightpink", "sandybrown", "thistle2","navajowhite", "palegreen2", "skyblue")
  
  plot <- ggplot(df, aes(x=reorder(cause_name, -prop), y=prop, fill=location_name, group=ID)) +
    geom_bar(position=position_dodge(), stat="identity", colour='black') +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(), axis.line = element_line(colour = "black"),
          axis.text.x = element_text(angle = 45, hjust = 1, size=22, color="black"),
          axis.text.y = element_text(size=22, color="black"),
          axis.title=element_text(size=24), legend.text=element_text(size=24),
          legend.title=element_text(size=24), plot.title = element_text(size=26)) + scale_x_discrete(labels = function(x) str_wrap(x, width = 25)) + 
    xlab("Cause Name") + labs(fill="Location Name") + ylab(paste(toupper(int_cause), "proportion")) + 
    scale_fill_manual(values = colors) + scale_y_continuous(expand = c(0, 0)) + ggtitle("Top 5 Causes") +
    guides(fill=guide_legend(keywidth=0.3, keyheight=0.5, default.unit="inch"))

  dir.create(paste0("/home/j/temp/agesak/thesis/figures/", DATE, "/"), showWarnings = FALSE)
  ggsave(paste0("/home/j/temp/agesak/thesis/figures/", DATE, "/", DATE, "_", int_cause, "_", short_name, "_by_country.pdf"), plot, dpi=300, height=12, width=22)
  
}

for (int_cause in c("x59", "y34")){
  for (short_name in names(CLASSIFIERS)){
    print(paste("working on", int_cause, short_name))
    plot_data(int_cause, short_name) 
  }
}
