#!/bin/bash
for FILE in $(ls /data/renamed_data/*.rtf); do 
    f="$(basename -- $FILE)"
    echo /data/renamed_data/$FILE
    echo /data/text_data_2/$f.txt;

    unrtf --text $FILE > /data/txt_data/$f.txt; 
    #echo $FILE.txt;
done