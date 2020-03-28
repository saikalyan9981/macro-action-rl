import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import json 
## add arguments
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    pullData = open("trace.txt","r").read()
    dataArray = pullData.split('\n')
    xar = []
    yar = []
    # info_dict ={}
    for eachLine in dataArray:
        if len(eachLine)>1:
            eachLine = eachLine.split(",")
            y = float(eachLine[2].split(":")[1])
            x = int(eachLine[0].split(":")[1])
            xar.append(x)
            yar.append(y)
    ax1.clear()
    ax1.plot(xar,yar)
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()