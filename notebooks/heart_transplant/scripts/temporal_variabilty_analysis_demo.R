
dataset <- read.csv2( "http://github.com/hms-dbmi/EHRtemporalVariability-DataExamples/raw/master/nhdsSubset.csv",
                      sep  = ",",
                      header = TRUE,
                      na.strings = "",
                      colClasses = c( "character", "numeric", "factor",
                                      "numeric" , rep( "factor", 22 ) ) )

datasetFormatted <- EHRtemporalVariability::formatDate(
              input         = dataset,
              dateColumn    = "date",
              dateFormat = "%y/%m"
             )

datasetPheWAS <- icd9toPheWAS(data           = datasetFormatted,
                              icd9ColumnName = "diagcode1",
                              phecodeDescription = TRUE,
                              missingValues  = "N/A",
                              statistics     = TRUE,
                              replaceColumn  = FALSE)

probMaps <- estimateDataTemporalMap(data           = datasetPheWAS,
                                    dateColumnName = "date",
                                    period         = "month")

igtProj <- estimateIGTProjection( dataTemporalMap = probMaps[[1]],
                                  dimensions      = 3,
                                  startDate       = "2000-01-01",
                                  endDate         = "2010-12-31")

plotDataTemporalMap(
    dataTemporalMap =  probMaps[["diagcode1-phewascode"]],
    startValue = 2,
    endValue = 20,
    colorPalette    = "Spectral")

plotIGTProjection(
    igtProjection   =  igtProj,
    colorPalette    = "Spectral",
    dimensions      = 3)
