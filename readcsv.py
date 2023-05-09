import pandas as pd

#adding headers to gfg.csv and gfgtest.csv
import csv
def writerheader(filename,header_content,newfilename):
    with open(filename+".csv", "r") as csv_file:
        # read the contents of the file
        reader = csv.reader(csv_file)
        # convert to list
        data = list(reader)
    # add the header to the list created above
    header_ls = header_content
    data.insert(0, header_ls)

    with open(newfilename+".csv", "w") as csv_file:
        # writer object
        writer = csv.writer(csv_file)
        # write data to the csv file
        writer.writerows(data)

writerheader("g4g",["video_name","tag"],"g4g1")
title = pd.read_csv("g4g1.csv")

writerheader("g4gtest",["video_name","tag"],"g4g1test")



print(title.tail())
