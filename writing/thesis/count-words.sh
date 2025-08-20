#!/bin/sh 

texcount -brief *.tex | awk '{ print $1, "\t", $4 }' | tr '+' '\t'
