import matplotlib

from Classifier import *
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tkinter as tk

import threading
import time


X = 50
Y = 50
HEIGHT = 10
WIDTH = 10

class Window(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self,master)
        master.geometry("950x600")
        self.createWidgets(master)

    def createWidgets(self, master):
        
        self.classifier = Classifier()
        self.classifier.build(input_shape=(1, 28*28))
        self.classifier.load_weights(f"./saved_models/trained_weights_50").expect_partial()
        
        self.classifier.summary()

        # Plot
        fig = plt.figure(figsize=(4,4))
       
        self.canvas_plot = FigureCanvasTkAgg(fig,master=master)
        self.canvas_plot.get_tk_widget().grid(row=0,column=1)


        self.canvas_field = tk.Canvas(master, height=500, width=500)
        self.elements = [[0 for x in range(28)] for i in range(28)]
        self.field = np.zeros(shape=(28,28))

        for i in range(28):
            for j in range(28):
                x0 = X + i*WIDTH
                y0 = Y + j*HEIGHT
                x1 = x0 + WIDTH
                y1 = y0 + HEIGHT
                elem = self.canvas_field.create_rectangle(x0 , y0, x1, y1,
                    outline="black", fill="white", tags="field")

                self.elements[i][j] = elem

                # Left button pressed
                self.canvas_field.tag_bind("field", "<B1-Motion>", self.fill)

                # Right button pressed
                self.canvas_field.tag_bind("field", "<B3-Motion>", self.removeFill)

        self.canvas_field.grid(row=0,column=0)

        self.plotbutton = tk.Button(master=master, text="Clear", command=self.click_button_eraseAll)
        self.plotbutton.grid(row=1,column=0)

        self.classify_thread = threading.Thread(target=self.classify)
        self.classify_thread.setDaemon(True)
        self.classify_thread.start()


    def click_button_eraseAll(self):

        for i in range(28):
            for j in range(28):
                box = self.elements[i][j]
                self.canvas_field.itemconfig(box, fill="white")
                self.field[i][j] = 0

    
    def classify(self):
     
        while True:
            # Clear data from old figure
            plt.clf()

            # add batch dim
            image = np.expand_dims(self.field, axis=0)
            image = tf.convert_to_tensor(image, np.float32)
            image = tf.reshape(image, (1,784))
            prediction = self.classifier(image)
            # remove batch dim
            prediction = prediction.numpy()
            prediction = prediction[0] 
            x = [str(i) for i in range(10)]  
            barPlot = plt.bar(x,prediction)

            idxMaxProb = np.argmax(prediction)
            barPlot[idxMaxProb].set_color('r')

            plt.xlabel("Digit")
            plt.ylabel("Probability")

            self.canvas_plot.draw()
            time.sleep(0.75)

    def fill(self, event):
        (x_idx, y_idx) = self.getBoxIndex(event.x, event.y)
        if x_idx == -1 or y_idx == -1:
            return

        box = self.elements[x_idx][y_idx]
        self.field[x_idx][y_idx] = 1
        self.canvas_field.itemconfig(box, fill="black")

    def removeFill(self, event):
        (x_idx, y_idx) = self.getBoxIndex(event.x, event.y)
        if x_idx == -1 or y_idx == -1:
            return
        box = self.elements[x_idx][y_idx]
        self.field[x_idx][y_idx] = 0
        self.canvas_field.itemconfig(box, fill="white")

    def getBoxIndex(self, x,y):
        x_idx = int( (x- X) / WIDTH )
        y_idx = int( (y - Y) / HEIGHT )
        if x_idx > 27 or x_idx < 0 :
            x_idx = -1

        if y_idx > 27 or y_idx < 0 :
            y_idx = -1

        return (x_idx, y_idx)
