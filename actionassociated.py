#segregating the label for train videos and creating csv file by name gfg.csv


import csv
import os
import pandas as pd
import time

st=time.process_time()
print(st)
direct=os.listdir("content/Hollywood2_final/videos/train")
direct_label=os.listdir("content/Hollywood2_final/labels/train")

Vi=(os.listdir("content/Hollywood2_final/videos/train"))
Vid=list(Vi)
Vid_labels=[]
#print(Vid)
#print(len(Vid))


#print(direct_label)

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

action=pd.read_csv("content/Hollywood2_final/labels/train/actions.txt")
element=action["actionindex"]

iaction=None
num,i=0,0

for num, em in enumerate(list(element)):
				#print(str(em))
				print(num)
				if (str(em[-1])!="-1"):
								for label in direct_label:
												if (label != "actions.txt"):
																dir = pd.read_csv(str("content/Hollywood2_final/labels/train/") + label, header=0)
																direct = dir.to_csv("pd.csv", header=None)
																with open('pd.csv', 'rb') as csvfile:
																				reader = csv.reader(open("pd.csv", 'r'))
																for score in (reader):
																				x = str(score)[-4:-2]

																				if (int(str(score)[-4:-2]) > 0):
																								num = num + 1
																								if (i < len(Vid) - 1):
																												for v in Vid:

																																if (str(score)[9:29] + ".avi" == v):
																																				Vid_labels.append([v, label.replace(".txt", "")])

								# print(Vid_labels)
								file = open('g4g.csv', 'w+', newline='')

								# writing the data into the file
								with file:
												write = csv.writer(file)
												write.writerows(Vid_labels)
et=time.process_time()
print(et-st)




