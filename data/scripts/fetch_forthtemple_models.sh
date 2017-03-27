#!/bin/bash

#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd data #/ $DIR

FILE=forthtemple_faster_rcnnmasgv_iter_20000.caffemodel
URL=https://www.dropbox.com/s/4k7ppr720pdao3j/$FILE


if [ -f $FILE ]; then
  echo "File already exists. "
 
fi

echo "Downloading pretrained Forthtemple model..."

wget $URL -O $FILE

echo "Done."
