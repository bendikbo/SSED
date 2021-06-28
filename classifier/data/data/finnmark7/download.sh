#! /bin/bash

echo ""
echo "#####################################"
echo "#     DOWNLOADING TRAIN DATASET     #"
echo "#####################################"
echo ""

fileId=1xJ0xcjjKF9lACeCcHj7UkqOYOTTw-gdf
fileName=train.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}

echo ""
echo "#####################################"
echo "#     DOWNLOADING TEST DATASET      #"
echo "#####################################"
echo ""

fileId=1A7l4Jt3DoZa_zqWzyuz2GfC8I_T0PCLy
fileName=test.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 

echo ""
echo "#####################################"
echo "#   DOWNLOADING VALIDATION DATASET  #"
echo "#####################################"
echo ""

fileId=1v6_4Cq0FP0pQGsQFdS0dZEIdf9dFuf7-
fileName=validate.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 
