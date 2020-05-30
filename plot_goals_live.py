import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import json 
import sys

## add arguments
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
algo = str(sys.argv[1])

print("plotting ... ")

def animate(i):
    pullData = open("logs/" + algo + ".log","r").read()

    dataArray = pullData.split('\n')

    xar = []
    yar = []
    # info_dict ={}
    for eachLine in dataArray:
        if "EndOfTrial:" in eachLine:
            eachLine = eachLine.split(" ")
            # x = float(eachLine[2].split(":")[1])
            # y = int(eachLine[0].split(":")[1])
            x = int(eachLine[1])
            y = int(eachLine[3])
            xar.append(x)
            yar.append(y)

    xplot = []
    for i, v in enumerate(xar, 1):
        if i < 2000:
            xplot.append(1 - v/i)
        else:
            xplot.append(1 - (v - xar[i - 2000])/2000)

    ax1.clear()
    if len(yar) > 600 and len(xplot) > 600:
        ax1.plot(yar[600:],xplot[600:])
    # ax1.plot(yar,xplot)    
    ax1.set_title(algo)
    ax1.set_xlabel("episodes")
    # print(xplot[-1])

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()