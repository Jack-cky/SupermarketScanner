#!/bin/bash

# get instance name from argument
instance=$1

# declear target directory and image extension
dict="/Users/jackchan/Downloads"
ext=".jpg"

# create a directory storing images
mkdir -p $instance

# rename images into incremental digitals
idx=1
for file in $dict/*$ext
    do
        mv "$file" $dict/$instance/img_$idx$ext
        ((idx=idx+1))
    done
