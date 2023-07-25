#!/bin/bash

for i in non-burps/*.wav; do
  id=`basename $i | sed -re 's/^(.*?)\.wav$/\1/'`
  echo $id >> blacklist.txt
done