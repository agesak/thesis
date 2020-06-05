# Bar graph of % of injuries garbage codes that are x59/y34 by location year
# Figure 2
rm(list=ls())

library(data.table)
library(ggplot2)
library(stringr)

int_causes <- c("x59", "y34")
thesis_dir <- "/home/j/temp/agesak/thesis"

for (int_cause in int_causes){
  df <- fread(paste0(thesis_dir, "/tables/percent_", int_cause, ".csv"))
  df <- df[get(paste0("cause_", int_cause)) %like% paste0(int_cause),]
  df[, paste0("percent_", int_cause) := get(paste0("percent_", int_cause))*100]
  
  plot <- ggplot(df, aes(x=location_name, y=get(paste0("percent_", int_cause)))) + 
    geom_bar(position=position_dodge(), stat="identity", fill = "mediumaquamarine") + 
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(), axis.line = element_line(colour = "black")) + 
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size=28, color="black"),
          axis.text.y = element_text(size=28, color="black"),
          axis.title=element_text(size=26)) + xlab("Country Name") + 
    ylab(paste("%", toupper(int_cause))) + 
    scale_y_continuous(labels = function(x) paste0(x, "%"), expand = c(0, 0), limits = c(0, 60)) +
    scale_x_discrete(labels = function(x) str_wrap(x, width = 20)) + theme(plot.margin = unit(c(1,1,1,1), "cm"))
  
  ggsave(paste0(thesis_dir, "/figures/percent_", int_cause , ".pdf"), plot, dpi=300, height=10, width=10)
  ggsave(paste0(thesis_dir, "/figures/percent_", int_cause , ".jpeg"), plot, dpi=300, height=10, width=10)
}
