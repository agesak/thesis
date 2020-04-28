# Bar graph of % of injuries garbage codes that are x59/y34 by location year
rm(list=ls())

library(data.table)
library(ggplot2)

int_causes <- c("x59", "y34")
figure_dir <- "/home/j/temp/agesak/thesis/figures"

for (int_cause in int_causes){
  df <- fread(paste0(figure_dir, "/percent_", int_cause, ".csv"))
  df <- df[get(paste0("cause_", int_cause)) %like% paste0(int_cause),]
  df[, percent := percent*100]
  
  plot <- ggplot(df, aes(x=location_name, y=percent)) + 
    geom_bar(position=position_dodge(), stat="identity", fill = "cadetblue2") + 
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(), axis.line = element_line(colour = "black")) + 
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size=10)) + xlab("Country Name") + 
    ylab(paste("Percent injuries garbage that is", toupper(int_cause))) + 
    scale_y_continuous(labels = function(x) paste0(x, "%"), expand = c(0, 0))
  
  ggsave(paste0(figure_dir, "/percent_", int_cause , ".pdf"), plot, dpi=300, height=5, width=8)
}
