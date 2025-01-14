#For initial learning started with the Youtube tutorial: Matplotlib Tutorial link --> htttps://youtube.com/watch?v=wB9C0Mz9gSo

import matplotlib.pyplot as plt
#%matplotlib inline #relevant to jupyter notebook
import numpy as np
import pandas as pd
import random

class MatplotHowItWorks:
    def __init__(self):
        self.x = []
        self.y = []
        self.generateRandomPlotDate()

    def generateRandomPlotDate(self):
        self.x = np.linspace(0, 5, 10)  # generate 10 floats from 0 to 5
        self.y = self.x ** 2

    def simplePlot(self):
        plt.plot(self.x,self.y)
        plt.title('Matplot Simple Chart')
        plt.xlabel('x values')
        plt.ylabel('y values')
        plt.show()

    def printMultiplePlot(self):
        plt.subplot(1,2,1)
        plt.plot(self.x,self.y,'red')
        plt.subplot(1,2,2)
        plt.plot(self.x,self.y,'blue')
        plt.show()

    def figureObjects(self):
        fig_1 = plt.figure(figsize=(5,4), dpi = 100)
        axes_1 = fig_1.add_axes([0.1,0.1,0.9,0.9]) #left, bottom, height, width; represents the % of canvas we want to use
        axes_1.set_xlabel('x values')
        axes_1.set_ylabel('y values')
        axes_1.set_title('Figure Plot')
        axes_1.plot(self.x, self.y,label='x/x2')
        axes_1.plot(self.y, self.x, label='x2/x')
        axes_1.legend(loc=0) #location 0 auto assigns the best location to put the legend
        #upper right: 1, upper left: 2, lower left : 3, lower right: 4, or a x&y tupple

        axes_2=fig_1.add_axes([0.45,0.45,0.4,0.3])
        axes_2.set_xlabel('x values')
        axes_2.set_ylabel('y values')
        axes_2.set_title('Plot 2nd Figure Object')
        axes_2.plot(self.x,self.y,'green')
        #axes_2.plot(self.y,self.x,label='x2/x')
        #axes_2.legend(loc=0)
        axes_2.text(0,40,'Message Text')
        plt.show()

    def subPlots(self):
        fig_1, axes_1 = plt.subplots(figsize=(8,4),nrows=1, ncols=2)
        plt.tight_layout() # add space between subplots
        axes_1[1].set_title('plot 2')
        axes_1[1].set_xlabel('X')
        axes_1[1].set_ylabel('Y')
        axes_1[1].plot(self.x, self.y)
        plt.show()
    def appearences(self):
        fig_3 = plt.figure(figsize=(8,4))
        #axes_3 = fig_3.add_axes([0,0,1,1])
        axes_3 = fig_3.add_axes([0.1, 0.1, 0.9, 0.9])
        axes_3.plot(self.x, self.y, color='navy', alpha=0.75, lw=2, ls='-.', marker='o', markersize=7, markerfacecolor='orange',
                    markeredgecolor='yellow', markeredgewidth=4)
        #axes_3.set_xlim([0,3])
        #axes_3.set_ylim([0,8])
        axes_3.grid(True, color='0.6',dashes=(5,2,1,2)) #dashes= 5 dashes, 2 space, 1 point, 2 spaces
        axes_3.set_facecolor('#FAEBD7')
        axes_3.annotate('Peak',xy=(5,25),xytext=(3.5,25),arrowprops=dict(facecolor='black',shrink=0.05))
        axes_3.text(1,25,r'$\alpha \beta \sigma \omega \mu \pi \theta \epsilon \lambda$')
        axes_3.text(1, 16, r'$\delta_i \gamma^{ij} \sum_{i=0}^\infty x_i \frac{3}{4} $')
        axes_3.text(0, 21, r'$\frac{8 -\frac{x}{4}}{8} \sqrt{9} \sin(\pi) \sqrt[3]{8} \acute a \div$')
        axes_3.text(0, 8, r'$\bar a \hat a \tilde a \vec a \overline {a} \lim_{x \to 2} f(x) = 5$')
        axes_3.text(4, 8, r'$\geq \leq \ne$')
        axes_3.bar(self.x,self.y)
        fig_3.savefig('appearences')
        plt.show()

    def pandasDataFrame(self):
        ics_df = pd.read_csv('file.csv')
        ics_df = ics_df.sort_values(by='column_name')
        np_arr = ics_df.values
        self.x = np_arr[:,0]
        self.y=np_arr[:,1]

    def charts(self):
        arr_1 = np.random.randint(1,7,7000)
        arr_2 = np.random.randint(1,7,7000)
        arr_3 = arr_1 + arr_2
        ####histogram#####
        plt.hist(arr_3, bins=11, density=True, stacked=True)
        #other option possible orientation='horizontal'; cumulative=True, histtype='step'; color='orange'
        ####bar chart#####
        #plt.bar()

        plt.show()


mhw = MatplotHowItWorks()
#mhw.simplePlot()
#mhw.printMultiplePlot()
#mhw.figureObjects()
#mhw.subPlots()
#mhw.appearences()
mhw.charts()

#Note: Color coding--> m:magenta, c:cyan,k:black, w:white, y:yellow
#color="0.75" creates a 75% grey
#color = "#eeefff"
#color = "burlywood"
#https://en.wikipedia.org/wiki/Web_colors




