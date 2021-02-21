library(EHRtemporalVariability)


data <- read.csv('../cardiovascular-risk-data/data/UNOS_v3.csv', sep = ",", header = TRUE, na.strings = "")

X <- data[, !(names(data) %in% c('y'))]

X <- X[!is.na(X$tx_year),]

X$tx_year <- as.Date(as.character(X$tx_year), "%Y")

probMaps <- estimateDataTemporalMap(
  data = X,
  dateColumnName = "tx_year",
  period = "year",
  startDate = as.Date("1995-01-01")
)

igtProj <- estimateIGTProjection(
  dataTemporalMap = probMaps[[1]],
  dimensions = 2,
)

plotDataTemporalMap(
    dataTemporalMap =  probMaps[["bmi_calc"]],
    colorPalette    = "Spectral"
)

plotDataTemporalMap(
    dataTemporalMap =  probMaps[["creat_trr"]],
    colorPalette    = "Spectral",
    startValue = 2,
    endValue = 20,
)


igtProjs <- sapply ( probMaps, estimateIGTProjection )

plotIGTProjection(
    igtProjection   =  igtProjs[["age"]],
    colorPalette    = "Spectral",
    dimensions      = 2)
