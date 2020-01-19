import tkinter
import pandas
import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple
from PIL import Image
from PIL import ImageTk
from tkinter import *
from PIL import Image, ImageTk
import csv

x = pandas.read_csv('./rgb_two_dominant.csv', header=None, sep="|")

root = Tk()
count = 0

photoArray = []


def ClickFalse():
	global count
	count = count + 1
	label.configure(image=photoArray[count][0])
	label2.configure(image=photoArray[count][1])
	complimentaryArray= [make_tuple(x[0][count]), make_tuple(x[1][count]), 0]
	with open('/Users/zach/code/python-projects/deep-education/rgb_two_dominant_with_label.csv', 'a', newline='') as csvfile:
		compCSV = csv.writer(csvfile, delimiter='|')
		compCSV.writerow(complimentaryArray)
	print(complimentaryArray)
	print(False)
	print("Updated")


def ClickTrue():
	global count
	count = count + 1
	label.configure(image=photoArray[count][0])
	label2.configure(image=photoArray[count][1])
	complimentaryArray= [make_tuple(x[0][count]), make_tuple(x[1][count]), 1]
	with open('/Users/zach/code/python-projects/deep-education/rgb_two_dominant_with_label_better.csv', 'a', newline='') as csvfile:
		compCSV = csv.writer(csvfile, delimiter='|')
		compCSV.writerow(complimentaryArray)
	print(complimentaryArray)
	print(True)
	print("Updated")


for i in range(len(x)):
	pictureOne = ImageTk.PhotoImage(Image.new("RGB", (75, 75), make_tuple()))
	pictureTwo = ImageTk.PhotoImage(Image.new("RGB", (75, 75), make_tuple(x[1][i])))
	array = [pictureOne, pictureTwo]
	photoArray.append(array)

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

left = Label(labelframe)

button = Button(labelframe, padx=5, pady=5, text="Not", command=ClickFalse)
button.pack(side=RIGHT)

button = Button(labelframe, padx=5, pady=5, text="Complimentary", command=ClickTrue)
button.pack(side=LEFT)

left.pack()
root.bind('n', lambda event: ClickFalse())
root.bind('y', lambda event: ClickTrue())
root.mainloop()
