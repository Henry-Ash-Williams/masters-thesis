#!/bin/sh 

# USAGE: 
# $ ls *.tex | entr -c ./count-words.sh

texcount -brief *.tex | awk '{ print $1, "\t", $4 }' | tr '+' '\t'
