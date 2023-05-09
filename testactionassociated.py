#segregating the label for test videos and creating csv file by name gfgtest.csv

import csv
import os
import pandas as pd
import time

st=time.process_time()
print(st)
direct=os.listdir("content/Hollywood2_final/videos/test")
direct_label=os.listdir("content/Hollywood2_final/labels/test")

Vi=(os.listdir("content/Hollywood2_final/videos/test"))
Vid=list(Vi)
Vid_labels=[]

classes_label=["HandShake", "Run",
 "StandUp",
 "AnswerPhone",
"FightPerson",
 "HugPerson",
 "SitDown",
 "DriveCar",
 "GetOutCar",
 "Kiss",
 "SitUp"]

action=pd.read_csv("content/Hollywood2_final/labels/test/actions.txt")
element=action["actionindex"]

iaction=None
num,i=0,0

for num, em in enumerate(list(element)):
				print(num)
				if (str(em[-1])!="-1"):
								for label in direct_label:
												if (label != "actions.txt"):
																dir = pd.read_csv(str("content/Hollywood2_final/labels/test/") + label, header=0)
																direct = dir.to_csv("pdtest.csv", header=None)
																with open('pdtest.csv', 'rb') as csvfile:
																				reader = csv.reader(open("pdtest.csv", 'r'))
																for score in (reader):
																				x = str(score)[-4:-2]

																				if (int(str(score)[-4:-2]) > 0):
																								num = num + 1
																								if (i < len(Vid) - 1):
																												for v in Vid:

																																if (str(score)[9:29] + ".avi" == v):
																																				Vid_labels.append([v, label.replace(".txt", "")])

								file = open('g4gtest.csv', 'w+', newline='')

								# writing the data into the file
								with file:
												write = csv.writer(file)
												write.writerows(Vid_labels)
et=time.process_time()
print(et-st)




