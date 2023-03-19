#!/bin/bash

wget -P .tmp/ -c https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_10.json.gz
gzip -d .tmp/reviews_Books_10.json.gz
wget -P .tmp/ -c https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz
gzip -d .tmp/meta_Books.json.gz


