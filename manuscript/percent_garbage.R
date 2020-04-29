# Bar graph of % of injuries garbage codes that are x59/y34 by location year
rm(list=ls())

library(data.table)
library(ggplot2)

int_causes <- c("x59", "y34")
thesis_dir <- "/home/j/temp/agesak/thesis"

for (int_cause in int_causes){
  df <- fread(paste0(thesis_dir, "/tables/percent_", int_cause, ".csv"))
  df <- df[get(paste0("cause_", int_cause)) %like% paste0(int_cause),]
  df[, paste0("percent_", int_cause) := get(paste0("percent_", int_cause))*100]
  
  plot <- ggplot(df, aes(x=location_name, y=get(paste0("percent_", int_cause)))) + 
    geom_bar(position=position_dodge(), stat="identity", fill = "cadetblue2") + 
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(), axis.line = element_line(colour = "black")) + 
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size=10)) + xlab("Country Name") + 
    ylab(paste("%", toupper(int_cause))) + 
    scale_y_continuous(labels = function(x) paste0(x, "%"), expand = c(0, 0))
  
  ggsave(paste0(thesis_dir, "/figures/percent_", int_cause , ".pdf"), plot, dpi=300, height=6, width=8)
  ggsave(paste0(thesis_dir, "/figures/percent_", int_cause , ".jpeg"), plot, dpi=300, height=6, width=8)
}
