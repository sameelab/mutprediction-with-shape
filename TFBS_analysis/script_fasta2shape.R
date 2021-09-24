### script_fasta2shape.R
### Zian Liu
### 2/4/2021

## This script makes DNA shape references across all known TF binding sites


# Library
library(DNAshapeR)


# Define directories, inputs 1 and 2 from command line
Args <- commandArgs(trailingOnly = TRUE)
DirIn = Args[1]
DirOut = Args[2]
print("Input and output directories: ")
print(DirIn)
print(DirOut)

# The package doesn't have an 'All' option, define all shapes used
ShapeTypeList = c("HelT", "Rise", "Roll", "Shift", "Slide", "Tilt", "Buckle", 
                  "Opening", "ProT", "Shear", "Stagger", "Stretch", "MGW", 
                  "EP")


# Iterate through all files in DirIn
InFileList = list.files(path=DirIn, pattern="*.fa")

for (InFile in InFileList){
  # Create output name
  OutFile = gsub("fa", "csv", gsub("motifs", "shape", InFile))
  
  # Import the fastas and run shapes
  ShapeTF = data.frame()
  for (ShType in ShapeTypeList){
    # Get DNA shape
    TempDF <- getShape(paste(DirIn, InFile, sep = ""), shapeType = ShType)[[ShType]]
    # Remove all NA columns
    TempDF <- TempDF[, colSums(is.na(TempDF)) < 1]
    # Change column name
    colnames(TempDF) <- paste(ShType, 1:ncol(TempDF), sep="_")
    # Add data
    if (dim(ShapeTF)[1] == 0) {
      ShapeTF <- TempDF
    } else {
      ShapeTF = cbind(ShapeTF, TempDF)
    }
  }
  
  # Make an output
  write.csv(ShapeTF, file = paste(DirOut, OutFile, sep=""), row.names = FALSE)
}

# Done. 
