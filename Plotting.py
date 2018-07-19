import os
import sys
import tempfile
import argparse
import subprocess
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time

''' --- Plotting the Results --- '''
''' --- Plot for scalar fields c(x,y,t) --- '''
def plot(x,y,k,cmin,cmax,tit,i):
    plt.cla()
    plt.pcolormesh(x,y,k,vmin=cmin,vmax=cmax, cmap = 'jet')
    plt.colorbar()
    plt.xlabel('x (a.u.)')
    plt.ylabel('y (a.u.)')
    plt.axis('equal')
    plt.axis([np.amin(x), np.amax(x), np.amin(y), np.amax(y)])
    plt.title('%s(x,y,t=%i$\Delta$)'%(tit,i), fontsize=20)
    filename = "temp/CH-2D-%s%04d.png" % (tit,i)
    plt.savefig(filename)
    plt.close()
    
def plot2(x,y,y0,y1,tit,i):
    plt.cla()
    plt.xlabel('t (a.u.)')
    plt.ylabel('Interface Fraction (a.u.)')
    plt.axis('equal')
    print(plt.ylim())
    plt.axis((1.,70.,0.,100.))
    plt.plot(x,(y*100.))
    plt.title('%s' % (tit), fontsize=20)
    filename = "temp/CH-2D-%s%04d.png" % (tit,i)
    plt.savefig(filename)
    plt.close()

''' --- Plot for Vector fields v = jx, u = jy --- '''
def plotvec(X, Y, v, u, tit,i):
    plt.title('%s'%tit, fontsize=20)
    plt.xlabel('x (a.u.)')
    plt.ylabel('y (a.u.)')
    plt.axis('equal')
    plt.axis([np.amin(X), np.amax(X), np.amin(Y), np.amax(Y)])
    plt.quiver(X, Y, v, u, units='dots')
    filename = "temp/CH-2D-%s%04d.png" % (tit,i)
    plt.savefig(filename)
    plt.close()
    
def Lineplot(x, c, line ,tit, i):
    plt.cla()
    #plt.plot(np.linspace(0, 1, Nx),c[64][:],np.linspace(0, 1, Nx),c[:][64])
    plt.plot(x,c)
    plt.axis([np.amin(x), np.amax(x), np.amin(c), np.amax(c)])
    filename = "temp/CH-2D-%s%04d.png" % (tit,i)
    plt.savefig(filename)
    plt.close()

''' --- Plot a video of files --- '''
def video(file, out, i):
    FPS = 5.
    Output = 'video-%s.mp4' % (out)

    print('\nRunning ffmpeg:')
    if (i==1):
        ffmpeg_options = {
            #'-filter:v' : 'setpts=%f*PTS' % (FPS),
            #'-b:v' : '1000k',
            '-f': 'image2',
            '-r': '5.',
            '-i': 'temp/CH-2D-c%04d.png'
            }
    elif (i==2):
        ffmpeg_options = {
            #'-filter:v' : 'setpts=%f*PTS' % (FPS),
            #'-b:v' : '1000k',
            '-f': 'image2',
            '-r': '5.',
            '-i': 'temp/CH-2D-mutot%04d.png'
            }
    elif (i==3):
        ffmpeg_options = {
            #'-filter:v' : 'setpts=%f*PTS' % (FPS),
            #'-b:v' : '1000k',
            '-f': 'image2',
            '-r': '5.',
            '-i': 'temp/CH-2D-DiffFlux%04d.png'
            }
    elif (i==4):
        ffmpeg_options = {
            #'-filter:v' : 'setpts=%f*PTS' % (FPS),
            #'-b:v' : '1000k',
            '-f': 'image2',
            '-r': '5.',
            '-i': 'temp/CH-2D-Reaction Flux%04d.png'
            }

    command = ['ffmpeg']

    for key, value in ffmpeg_options.items():
         print((key, str(value)))
         command.extend((key, str(value)))

    command.append('-y')
    print(Output)
    command.append(Output)

    subprocess.call(command)

    print('\nWrote output to %s' % Output)
    
''' --- Remove a directory --- '''
def remove(dir_name):
    for file in os.listdir('temp'): 
         file_path = os.path.join(dir_name, file)
         if os.path.isfile(file_path):
             os.remove(file_path)
    os.rmdir(dir_name)