# Bar graph of % of injuries garbage codes that are x59/y34 by location year
library(data.table)
library(ggplot2)

int_causes <- c("x59", "y34")
int_cause <- "x59"
df <- fread(paste0("/home/j/temp/agesak/thesis/figures/percent_", int_cause, ".csv"))
# df <- fread("/homes/agesak/thesis_graph.csv")
df <-df[get(paste0("cause_", int_cause)) %like% paste0(int_cause),]

ggplot(df, aes(x=location_name, y=percent)) + 
  geom_bar(position=position_dodge(), stat="identity", colour='black') + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size=10)) + xlab("Country Name") + 
  ylab(paste("Percent", int_cause, "of injuries garbage"))
