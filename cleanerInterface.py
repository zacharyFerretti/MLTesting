# Tkinter imports.
import tkinter
from tkinter import *

# Data Processing Imports
import pandas
import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple

# Image Generation Imports
from PIL import Image
from PIL import ImageTk

import random
import csv

x = pandas.read_csv('./list_of_rgb.csv', header=None, sep="|")
print(x)
root = Tk()

count = 0

photoArray = []


def Click(theValue):
	global count


	label.configure(image=photoArray[count][0])
	label2.configure(image=photoArray[count][1])
	complimentaryArray = [colorArray[count][0], colorArray[count][1], intValue]
	with open('/Users/zach/code/python-projects/deep-education/rgb_labeled_explored.csv', 'a', newline='') as csvfile:
		compCSV = csv.writer(csvfile, delimiter='|')
		compCSV.writerow(complimentaryArray)
	count = count + 1
	print(complimentaryArray)
	print(theValue)
	print("Updated")


colorArray = []
for i in range(len(x)):
	xTuple = make_tuple(x[0][i])
	pictureOne = ImageTk.PhotoImage(Image.new("RGB", (150, 150), xTuple))
	for j in range(8):
		r = random.randint(0, 256)
		g = random.randint(0, 256)
		b = random.randint(0, 256)
		rgbTuple = (r, g, b)
		pictureTwo = ImageTk.PhotoImage(Image.new("RGB", (150, 150), rgbTuple))
		array = [pictureOne, pictureTwo]
		tupleArray = [xTuple, rgbTuple]
		colorArray.append(tupleArray)
		photoArray.append(array)

print(len(photoArray))
print(len(photoArray[0]))
print(type(photoArray[0][0]))
# photo = ImageTk.PhotoImage(Image.new("RGB", (75, 75), make_tuple(x[0][0])))
# photo2 = ImageTk.PhotoImage(Image.new("RGB", (75, 75), make_tuple(x[0][1])))

label = Label(root, image=photoArray[0][0])
label2 = Label(root, image=photoArray[0][1])

label.image = photoArray[0][0]  # keep a reference!
label.pack()

label2.image = photoArray[0][1]
label2.pack()

labelframe = LabelFrame(root)
labelframe.pack(fill="both", expand="yes")

label.configure(background='black')
label2.configure(background='black')
left = Label(labelframe)

# button = Button(labelframe, padx=5, pady=5, text="Not", command=Click(False))
# button.pack(side=RIGHT)

# button = Button(labelframe, padx=5, pady=5, text="Complimentary", command=Click(True))
# button.pack(side=LEFT)

left.pack()
root.bind('n', lambda event: Click(False))
root.bind('y', lambda event: Click(True))
root.mainloop()
