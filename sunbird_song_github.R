## set working directory
library(mclust)

##There will be four total matrices/data frames in this analysis - an element matrix, a song matrix, an individual matrix, and a population matrix.
data <- read.csv("sunbird_song_data.csv", header = 1)

##Output length of data to see how many observations (song elements) there are in the set.
length(data$Individual.name)
##38416 song elements

##output number of songs in data set
length(unique(data$Song.name))
##419 songs

##Output a vector of names of individuals in the data set.
Individual<-unique(data$Individual.name)
##Output number of individuals in data set
length(Individual)
##142 individuals

data$elementfreqrange<-data$Peak.frequency.Maximum-data$Peak.frequency.Minimum

##Make vectors of 1's that will then be altered to make dummy variables representing song and individual identifications. Add these to the matrix of element observations. 
dummy<-rep(c(1),length(data$Individual.name))
data<-cbind(data,dummy)
song<-rep(c(1),length(data$Individual.name))
data<-cbind(data,song)


##Make the vector dummy represent the individuals in the sample by assigning each individual a number for each row in the 'element matrix.'
i<-1;
while(i<=length(Individual)){
data$dummy[which(data$Individual.name==Individual[i])]<-i
i<-i+1
}

##Check the vector to see how many individuals there are.
unique(data$dummy)

##Create vector with all the song IDs in it (becomes a vector of song IDs in 'song matrix').
songs<-unique(data$Song.name)
length(songs) ##Check number of songs (419).
unique(songs) ##See names of each of 419 songs.

##Create a vector of 1's that will be converted to a vector of individual IDs by number for each song (to identify individuals within the 'song matrix').
individual.by.song<-rep(c(1),length(songs))

##In the 'element matrix', give each song an integer ID (each element has an integer ID which tells you the song that each element is in.)
i<-1;
while(i<=length(songs)){
data$song[which(data$Song.name==songs[i])]<-i
i<-i+1
}

##(For the 'song matrix') Replace the 1's in the individual.by.song vector with the appropriate integer value representing individual ID. Each song is then associated with an individual in this vector.
i<-1;
while(i<=length(songs)){
individual.by.song[i]<-data$dummy[which(data$song==i)]
i<-i+1
}
length(individual.by.song) ##Check the modified vector - 419 elements


##Factorize the song and dummy vectors in the 'element matrix.'
data$song<-as.factor(data$song)
data$dummy<-as.factor(data$dummy)

##Change null and negative values in Gap.after to NAs (coded as -10000 in Luscinia; negative values are from overlapping elements in Luscinia (~150 from 28405 observations (0.5%), disregarded))
data$Gap.after[which(data$Gap.after==-10000)]=NA
data$Gap.after[data$Gap.after<0]=NA

## Define function to calculate coefficient of variation from variables
co.var<-function(x)(100*sd(x)/mean(x))

##Extract variables by song.
median.gap.after<-tapply(data$Gap.after[data$Gap.after>=0],data$song[data$Gap.after>=0],median)
median.gap.after<-median.gap.after[]
median.peak.frequency<-tapply(data$Peak.frequency.Average..mean.,data$song,median)
median.peak.frequency<-median.peak.frequency[]
co.var.peak.frequency<-tapply(data$Peak.frequency.Average..mean.,data$song,co.var)
co.var.peak.frequency<-co.var.peak.frequency[]
#sd.peak.frequency <- tapply(data$Peak.frequency.Average..mean., data$song, sd)
#sd.peak.frequency <- sd.peak.frequency[]
max.peak.frequency<-tapply(data$Peak.frequency.Average..mean.,data$song,max)
max.peak.frequency<-max.peak.frequency[]
min.peak.frequency<-tapply(data$Peak.frequency.Average..mean.,data$song,min)
min.peak.frequency<-min.peak.frequency[]
range.peak.frequency<-max.peak.frequency-min.peak.frequency
range.peak.frequency<-range.peak.frequency[]
elements<-tapply(data$Element.Number,data$song,max)
elements<-elements[]
median.bandwidth<-tapply(data$Frequency.bandwidth.Average..mean.,data$song,median)
median.bandwidth<-median.bandwidth[]
co.var.bandwidth<-tapply(data$Frequency.bandwidth.Average..mean.,data$song,co.var)
co.var.bandwidth<-co.var.bandwidth[]
#sd.bandwidth <- tapply(data$Frequency.bandwidth.Average..mean.,data$song,sd)
#sd.bandwidth <- sd.bandwidth[]
co.var.gap.after<-tapply(data$Gap.after[data$Gap.after>=0],data$song[data$Gap.after>=0],co.var)
co.var.gap.after<-co.var.gap.after[]
#sd.gap.after <- tapply(data$Gap.after[data$Gap.after>=0],data$song[data$Gap.after>=0],sd)
#sd.gap.after <- sd.gap.after[]
median.frequency.change<-tapply(data$Abs..Peak.Frequency.Change.Average..mean.,data$song,median)
median.frequency.change<-median.frequency.change[]
co.var.frequency.change<-tapply(data$Abs..Peak.Frequency.Change.Average..mean.,data$song,co.var)
co.var.frequency.change<-co.var.frequency.change[]
#sd.frequency.change <- tapply(data$Abs..Peak.Frequency.Change.Average..mean.,data$song,sd)
#sd.frequency.change <- sd.frequency.change[]
median.element.duration<-tapply(data$Length,data$song,median)
median.element.duration<-median.element.duration[]
median.element.freqrange<-tapply(data$elementfreqrange,data$song,median)


##Duration calculation
start<-data$Start.time[which(data$Element.Number==1)] ##Make vector with start time in each song recording
last.syllable.start.time<-rep(c(1),length(songs)) ##Vector of 1's that will be replaced with start time of final syllable from each song

##procures start times of final syllable of each song, then puts them into last.syllable.start.time
i<-1
while(i<=length(songs)){
song.elements<-subset(data,data$song==i)
song.elements$Start.time[which.max(song.elements$Element.Number)]->last.syllable.start.time[i]
i<-i+1
}

Length.last.element<-rep(c(1),length(songs)) ##Make vector of 1's that will be replaced with duration of last element of each song

##procures durations of final syllables of each song, then puts them into last.syllable.start.time
i<-1
while(i<=length(songs)){
song.elements<-subset(data,data$song==i)
song.elements$Length[which.max(song.elements$Element.Number)]->Length.last.element[i]
i<-i+1
}

end<-last.syllable.start.time+Length.last.element ##Calculate end time of each song

Duration<-end-start ## Calculate Duration from end and start times
length(Duration) ##Check length
Duration ##Examine data
hist(Duration)


song.matrix<-cbind(median.gap.after,co.var.gap.after,median.peak.frequency,co.var.peak.frequency,max.peak.frequency,min.peak.frequency,range.peak.frequency,elements,median.bandwidth,co.var.bandwidth,median.frequency.change,co.var.frequency.change,Duration,median.element.duration,median.element.freqrange)

##Convert song matrix to data frame
song.df<-as.data.frame(song.matrix)
song.df<-cbind(songs,individual.by.song,song.df)

song.df$Duration==Duration

song.df$individual.by.song<-as.factor(song.df$individual.by.song)

##Take means, maxes, or mins, of all song variables across individuals for the 'individual matrix'
ind.median.gap.after<-tapply(song.df$median.gap.after,song.df$individual.by.song,mean)
ind.median.gap.after<-ind.median.gap.after[]
ind.co.var.gap.after<-tapply(song.df$co.var.gap.after,song.df$individual.by.song,mean)
ind.co.var.gap.after<-ind.co.var.gap.after[]
ind.median.peak.frequency<-tapply(song.df$median.peak.frequency,song.df$individual.by.song,mean)
ind.median.peak.frequency<-ind.median.peak.frequency[]
ind.co.var.peak.frequency<-tapply(song.df$co.var.peak.frequency,song.df$individual.by.song,mean)
ind.co.var.peak.frequency<-ind.co.var.peak.frequency[]
ind.max.peak.frequency<-tapply(song.df$max.peak.frequency,song.df$individual.by.song,mean)
ind.max.peak.frequency<-ind.max.peak.frequency[]
ind.min.peak.frequency<-tapply(song.df$min.peak.frequency,song.df$individual.by.song,mean)
ind.min.peak.frequency<-ind.min.peak.frequency[]
ind.range.peak.frequency<-tapply(song.df$range.peak.frequency,song.df$individual.by.song,mean)
ind.range.peak.frequency<-ind.range.peak.frequency[]
ind.elements<-tapply(song.df$elements,song.df$individual.by.song,mean)
ind.elements<-ind.elements[]
ind.median.bandwidth<-tapply(song.df$median.bandwidth,song.df$individual.by.song,mean)
ind.median.bandwidth<-ind.median.bandwidth[]
ind.co.var.bandwidth<-tapply(song.df$co.var.bandwidth,song.df$individual.by.song,mean)
ind.co.var.bandwidth<-ind.co.var.bandwidth[]
ind.median.frequency.change<-tapply(song.df$median.frequency.change,song.df$individual.by.song,mean)
ind.median.frequency.change<-ind.median.frequency.change[]
ind.co.var.frequency.change<-tapply(song.df$co.var.frequency.change,song.df$individual.by.song,mean)
ind.co.var.frequency.change<-ind.co.var.frequency.change[]
ind.Duration<-tapply(song.df$Duration,song.df$individual.by.song,mean)
ind.Duration<-ind.Duration[]
ind.median.element.duration<-tapply(song.df$median.element.duration,song.df$individual.by.song,mean)
ind.median.element.duration<-ind.median.element.duration[]
ind.median.element.freqrange<-tapply(song.df$median.element.freqrange,song.df$individual.by.song,mean)
ind.median.element.freqrange<-ind.median.element.freqrange[]


log.ind.elements<-log(ind.elements)
log.ind.median.bandwidth<-log(ind.median.bandwidth)
log.ind.median.frequency.change<-log(ind.median.frequency.change)
log.duration<-log(ind.Duration)

##Create ind.matrix for values at individual level
ind.matrix<-cbind(ind.median.gap.after,ind.co.var.gap.after,ind.median.peak.frequency,ind.co.var.peak.frequency,ind.max.peak.frequency,ind.min.peak.frequency,ind.range.peak.frequency,log.ind.elements,log.ind.median.bandwidth,ind.co.var.bandwidth,log.ind.median.frequency.change,ind.co.var.frequency.change,log.duration,ind.median.element.duration)
##Alternate individual matrix including median element frequency range
#ind.matrix<-cbind(ind.median.gap.after,ind.co.var.gap.after,ind.median.peak.frequency,ind.co.var.peak.frequency,ind.max.peak.frequency,ind.min.peak.frequency,ind.range.peak.frequency,log.ind.elements,log.ind.median.bandwidth,ind.co.var.bandwidth,log.ind.median.frequency.change,ind.co.var.frequency.change,log.duration,ind.median.element.duration, ind.median.element.freqrange)

ind.matrix<-as.data.frame(ind.matrix)
ind.df<-as.data.frame(ind.matrix,row.names=Individual)

#############################################################
#Generate data for comp method analysis
species_and_pops <- read.csv("species_and_pops_for_comp_method.csv")

ind.df.pops <- ind.df 
ind.df.pops$pop <- as.character(species_and_pops$pop)

gene_pops <- c("Ikokoto", "Image", "Mafwemiro", "Mamirwa", "Mazumbai", "Mbulu", "MtKenya", "Mufindi", "Namuli", "Ndundulu", "Nguru", "Njesi", "Rungwe", "Selebu", "Shengena", "Uluguru")

ind.df.pops.g <- ind.df.pops[ind.df.pops$pop %in% gene_pops,]

ind.df.pops.g$pop[which(ind.df.pops.g$pop == "Shengena")] <- "SouthPare"

##var.s is the unbiased estimator of population variance
var.s <- function(x) sum((x - mean(x))^2)/length(x-1)

ind.median.gap.after.pop <- tapply(ind.df.pops.g$ind.median.gap.after, ind.df.pops.g$pop, FUN = mean)
ind.median.gap.after.pop.var <- tapply(ind.df.pops.g$ind.median.gap.after, ind.df.pops.g$pop, FUN = var.s)
ind.median.gap.after.pop <- data.frame(ind.median.gap.after.pop, ind.median.gap.after.pop.var)
write.table(ind.median.gap.after.pop, file = "ind.median.gap.after.pop.csv", row.names = TRUE, col.names = FALSE, sep = ",", quote = FALSE)

ind.co.var.gap.after.pop <- tapply(ind.df.pops.g$ind.co.var.gap.after, ind.df.pops.g$pop, FUN = mean)
ind.co.var.gap.after.pop.var <- tapply(ind.df.pops.g$ind.co.var.gap.after, ind.df.pops.g$pop, FUN = var.s)
ind.co.var.gap.after.pop <- data.frame(ind.co.var.gap.after.pop, ind.co.var.gap.after.pop.var)
write.table(ind.co.var.gap.after.pop, file = "ind.co.var.gap.after.pop.csv", row.names = TRUE, col.names = FALSE, sep = ",", quote = FALSE)

ind.median.peak.frequency.pop <- tapply(ind.df.pops.g$ind.median.peak.frequency, ind.df.pops.g$pop, FUN = mean)
ind.median.peak.frequency.pop.var <- tapply(ind.df.pops.g$ind.median.peak.frequency, ind.df.pops.g$pop, FUN = var.s)
ind.median.peak.frequency.pop <- data.frame(ind.median.peak.frequency.pop, ind.median.peak.frequency.pop.var)
write.table(ind.median.peak.frequency.pop, file = "ind.median.peak.frequency.pop.csv", row.names = TRUE, col.names = FALSE, sep = ",", quote = FALSE)

ind.co.var.peak.frequency.pop <- tapply(ind.df.pops.g$ind.co.var.peak.frequency, ind.df.pops.g$pop, FUN = mean)
ind.co.var.peak.frequency.pop.var <- tapply(ind.df.pops.g$ind.co.var.peak.frequency, ind.df.pops.g$pop, FUN = var.s)
ind.co.var.peak.frequency.pop <- data.frame(ind.co.var.peak.frequency.pop, ind.co.var.peak.frequency.pop.var)
write.table(ind.co.var.peak.frequency.pop, file = "ind.co.var.peak.frequency.pop.csv", row.names = TRUE, col.names = FALSE, sep = ",", quote = FALSE)

ind.max.peak.frequency.pop <- tapply(ind.df.pops.g$ind.max.peak.frequency, ind.df.pops.g$pop, FUN = mean)
ind.max.peak.frequency.pop.var <- tapply(ind.df.pops.g$ind.max.peak.frequency, ind.df.pops.g$pop, FUN = var.s)
ind.max.peak.frequency.pop <- data.frame(ind.max.peak.frequency.pop, ind.max.peak.frequency.pop.var)
write.table(ind.max.peak.frequency.pop, file = "ind.max.peak.frequency.pop.csv", row.names = TRUE, col.names = FALSE, sep = ",", quote = FALSE)

ind.min.peak.frequency.pop <- tapply(ind.df.pops.g$ind.min.peak.frequency, ind.df.pops.g$pop, FUN = mean)
ind.min.peak.frequency.pop.var <- tapply(ind.df.pops.g$ind.min.peak.frequency, ind.df.pops.g$pop, FUN = var.s)
ind.min.peak.frequency.pop <- data.frame(ind.min.peak.frequency.pop, ind.min.peak.frequency.pop.var)
write.table(ind.min.peak.frequency.pop, file = "ind.min.peak.frequency.pop.csv", row.names = TRUE, col.names = FALSE, sep = ",", quote = FALSE)

ind.range.peak.frequency.pop <- tapply(ind.df.pops.g$ind.range.peak.frequency, ind.df.pops.g$pop, FUN = mean)
ind.range.peak.frequency.pop.var <- tapply(ind.df.pops.g$ind.range.peak.frequency, ind.df.pops.g$pop, FUN = var.s)
ind.range.peak.frequency.pop <- data.frame(ind.range.peak.frequency.pop, ind.range.peak.frequency.pop.var)
write.table(ind.range.peak.frequency.pop, file = "ind.range.peak.frequency.pop.csv", row.names = TRUE, col.names = FALSE, sep = ",", quote = FALSE)

log.ind.elements.pop <- tapply(ind.df.pops.g$log.ind.elements, ind.df.pops.g$pop, FUN = mean)
log.ind.elements.pop.var <- tapply(ind.df.pops.g$log.ind.elements, ind.df.pops.g$pop, FUN = var.s)
log.ind.elements.pop <- data.frame(log.ind.elements.pop, log.ind.elements.pop.var)
write.table(log.ind.elements.pop, file = "log.ind.elements.pop.csv", row.names = TRUE, col.names = FALSE, sep = ",", quote = FALSE)

log.ind.median.bandwidth.pop <- tapply(ind.df.pops.g$log.ind.median.bandwidth, ind.df.pops.g$pop, FUN = mean)
log.ind.median.bandwidth.pop.var <- tapply(ind.df.pops.g$log.ind.median.bandwidth, ind.df.pops.g$pop, FUN = var.s)
log.ind.median.bandwidth.pop <- data.frame(log.ind.median.bandwidth.pop, log.ind.median.bandwidth.pop.var)
write.table(log.ind.median.bandwidth.pop, file = "log.ind.median.bandwidth.pop.csv", row.names = TRUE, col.names = FALSE, sep = ",", quote = FALSE)

ind.co.var.bandwidth.pop <- tapply(ind.df.pops.g$ind.co.var.bandwidth, ind.df.pops.g$pop, FUN = mean)
ind.co.var.bandwidth.pop.var <- tapply(ind.df.pops.g$ind.co.var.bandwidth, ind.df.pops.g$pop, FUN = var.s)
ind.co.var.bandwidth.pop <- data.frame(ind.co.var.bandwidth.pop, ind.co.var.bandwidth.pop.var)
write.table(ind.co.var.bandwidth.pop, file = "ind.co.var.bandwidth.pop.csv", row.names = TRUE, col.names = FALSE, sep = ",", quote = FALSE)

log.ind.median.frequency.change.pop <- tapply(ind.df.pops.g$log.ind.median.frequency.change, ind.df.pops.g$pop, FUN = mean)
log.ind.median.frequency.change.pop.var <- tapply(ind.df.pops.g$log.ind.median.frequency.change, ind.df.pops.g$pop, FUN = var.s)
log.ind.median.frequency.change.pop <- data.frame(log.ind.median.frequency.change.pop, log.ind.median.frequency.change.pop.var)
write.table(log.ind.median.frequency.change.pop, file = "log.ind.median.frequency.change.pop.csv", row.names = TRUE, col.names = FALSE, sep = ",", quote = FALSE)

ind.co.var.frequency.change.pop <- tapply(ind.df.pops.g$ind.co.var.frequency.change, ind.df.pops.g$pop, FUN = mean)
ind.co.var.frequency.change.pop.var <- tapply(ind.df.pops.g$ind.co.var.frequency.change, ind.df.pops.g$pop, FUN = var.s)
ind.co.var.frequency.change.pop <- data.frame(ind.co.var.frequency.change.pop, ind.co.var.frequency.change.pop.var)
write.table(ind.co.var.frequency.change.pop, file = "ind.co.var.frequency.change.pop.csv", row.names = TRUE, col.names = FALSE, sep = ",", quote = FALSE)

log.duration.pop <- tapply(ind.df.pops.g$log.duration, ind.df.pops.g$pop, FUN = mean)
log.duration.pop.var <- tapply(ind.df.pops.g$log.duration, ind.df.pops.g$pop, FUN = var.s)
log.duration.pop <- data.frame(log.duration.pop, log.duration.pop.var)
write.table(log.duration.pop, file = "log_duration_by_pop.csv", row.names = TRUE, col.names = FALSE, sep = ",", quote = FALSE)

ind.median.element.duration.pop <- tapply(ind.df.pops.g$ind.median.element.duration, ind.df.pops.g$pop, FUN = mean)
ind.median.element.duration.pop.var <- tapply(ind.df.pops.g$ind.median.element.duration, ind.df.pops.g$pop, FUN = var.s)
ind.median.element.duration.pop <- data.frame(ind.median.element.duration.pop, ind.median.element.duration.pop.var)
write.table(ind.median.element.duration.pop, file = "ind.median.element.duration.pop.csv", row.names = TRUE, col.names = FALSE, sep = ",", quote = FALSE)

#################################################################################################

##Prior to clustering analyses, remove individual "Point 128, Nyumbanitu" from the data set (genetically moreaui individual from hybrid zone with aberrant "mixed" moreaui-fuelleborni song). That reduces the data set size to 141 individuals, based on measurements from 415 songs and 38,178 elements
ind.df <- ind.df[which(rownames(ind.df) != "Point 128, Nyumbanitu"),]

mc_songs <- Mclust(ind.df, modelNames = c("EII", "VII", "EEI", "EVI", "VEI", "VVI"))
plot(mc_songs)
summary(mc_songs)
classes <- as.data.frame(mc_songs$classification)
colnames(classes)<-c("classes")
#head(classes)
species_and_pops <- read.csv("species_and_pops_for_mclust_noPt28Nyu.csv")
classes$species<-species_and_pops$sp
class_errors <- classError(classes$classes,classes$species)
misclassed_inds <- Individual[class_errors$misclassified]

png(filename = "FigureS2.png", width = 8, height = 8, units = "in", res = 300)
clPairs(data = ind.df[,c(1,2,3,4,8, 13)], classification = classes$species)
dev.off()

songDR <- MclustDR(mc_songs)

summary(songDR)
plot(songDR)
plot(songDR,what="evalues")
plot(songDR,dimens=2:3)