#! /bin/bash

echo ""
echo "#####################################"
echo "#     DOWNLOADING TRAIN DATASET     #"
echo "#####################################"
echo ""

fileId=1ZrB4lwnU6ZN9fBYDN2DLG-6RO30RgApY
fileName=train.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}

echo ""
echo "#####################################"
echo "#     DOWNLOADING TEST DATASET      #"
echo "#####################################"
echo ""

fileId=1U-SkL4FtYEKzJaltNz-ymuuDc1S38hX1
fileName=test.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 

echo ""
echo "#####################################"
echo "#   DOWNLOADING VALIDATION DATASET  #"
echo "#####################################"
echo ""

fileId=1chB-cxYPrTNiKzX7lCWx_37-tjYzRfGQ
fileName=validate.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 
