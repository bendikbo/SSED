#! /bin/bash

echo ""
echo "#####################################"
echo "#     DOWNLOADING TRAIN DATASET     #"
echo "#####################################"
echo ""

fileId=1ZVUeOujtDvzbJ63WX3c-yn2flsJzrPV_
fileName=train.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}

echo ""
echo "#####################################"
echo "#     DOWNLOADING TEST DATASET      #"
echo "#####################################"
echo ""

fileId=1iPg1cpAbsJqjGWCRqMdZuzgZMzyaP4PI
fileName=test.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 

echo ""
echo "#####################################"
echo "#   DOWNLOADING VALIDATION DATASET  #"
echo "#####################################"
echo ""

fileId=1Ert2aFaz8_mIpQ2bz7qZzbEP8Vh0gR86
fileName=validate.zip
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 
