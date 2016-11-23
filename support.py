import numpy as np
import matplotlib
matplotlib.use('Agg')  # must be done on cluster as there is no display
from brian2 import *
import struct
import os

V3 = "aedat3"
V2 = "aedat"  # current 32bit file format
V1 = "dat"  # old format

EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event

RESX = 128
RESY = 128
RES = RESX * RESY
COLOURS = ['r', 'g', 'b', 'm', 'c', 'm', 'y', 'k']

##      Some helpful functions
flatIndex = lambda x, y : y*RESX + x
squareIndex = lambda i : (i % 128, int(i / 128))  # TODO there is probably some math todo for DAVIS 



def loadaerdat(datafile='/tmp/aerout.dat', length=0, version=V2, debug=1, camera='DVS128'):
    """    
    load AER data file and parse these properties of AE events:
    - timestamps (in us), 
    - x,y-position [0..127]
    - polarity (0/1)

    @param datafile - path to the file to read
    @param length - how many bytes(B) should be read; default 0=whole file
    @param version - which file format version is used: "aedat" = v2, "dat" = v1 (old)
    @param debug - 0 = silent, 1 (default) = print summary, >=2 = print all debug
    @param camera='DVS128' or 'DAVIS240'
    @return (ts, xpos, ypos, pol) 4-tuple of lists containing data of all events;
    """
    # constants
    aeLen = 8  # 1 AE event takes 8 bytes
    readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
    td = 0.000001  # timestep is 1us   
    if(camera == 'DVS128'):
        xmask = 0x00fe
        xshift = 1
        ymask = 0x7f00
        yshift = 8
        pmask = 0x1
        pshift = 0
    elif(camera == 'DAVIS240'):  # values take from scripts/matlab/getDVS*.m
        xmask = 0x003ff000
        xshift = 12
        ymask = 0x7fc00000
        yshift = 22
        pmask = 0x800
        pshift = 11
        eventtypeshift = 31
    else:
        raise ValueError("Unsupported camera: %s" % (camera))

    if (version == V1):
        print ("using the old .dat format")
        aeLen = 6
        readMode = '>HI'  # ushot, ulong = 2B+4B

    aerdatafh = open(datafile, 'rb')
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)
    if length == 0:
        length = statinfo.st_size    
    print ("file size", length)
    
    # header
    lt = aerdatafh.readline().decode('utf-8')
    while lt and lt[0] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline().decode('utf-8') 
        if lt.startswith('#End of Preferences'):
            p += len(lt)   # TODO this is a bit lazy, rearrange logic
            k += 1
            break
        if debug >= 2:
            #print (str(lt))
            pass
        continue
    
    # variables to parse
    timestamps = []
    xaddr = []
    yaddr = []
    pol = []
    
    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen
    length += p
    
    #print (xmask, xshift, ymask, yshift, pmask, pshift, aeLen,  p, length)    
    while p < length:
        #print (xmask, xshift, ymask, yshift, pmask, pshift, aeLen,  p, length)
        #ii = int.from_bytes(s, byteorder='big')
        #print(str(bin(ii)), len(str(bin(ii))))
        #print(type(s))
        addr, ts = struct.unpack(readMode, s)
        
        # parse event type
        if(camera == 'DAVIS240'):
            eventtype = (addr >> eventtypeshift)
        else:  # DVS128
            eventtype = EVT_DVS
        
        # parse event's data
        if(eventtype == EVT_DVS):  # this is a DVS event
            #print(str(bin(addr)), len(str(bin(addr))))
            x_addr = (addr & xmask) >> xshift
            y_addr = (addr & ymask) >> yshift
            a_pol = (addr & pmask) >> pshift


            if debug >= 3: 
                print("ts->", ts)  # ok
                print("x-> ", x_addr)
                print("y-> ", y_addr)
                print("pol->", a_pol)

            timestamps.append(ts)
            xaddr.append(x_addr)
            yaddr.append(y_addr)
            pol.append(a_pol)
            #poldf
        
        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen        

    if debug > 0:
        print(len(timestamps))
        try:
            print ("read %i (~ %.2fM) AE events, duration= %.2fs" % (len(timestamps), len(timestamps) / float(10 ** 6), (timestamps[-1] - timestamps[0]) * td))
            n = 5
            print ("showing first %i:" % (n))
            print ("timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s" % (timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
        except Exception as e:
            print(e)
            print ("failed to print statistics")

    return timestamps, xaddr, yaddr, pol


def rec_field(ofx, ofy, sizex, sizey, stim_res_x=RESX, stim_res_y=RESY):
    """ Given the offset in (x,y) and a size in x and y return a receptive field
        as a list of indices assuming camera stimulus if not specified for stim_res_*
        The offset plus corresponding size will not excede the retina
    """
    if ofx + sizex >= stim_res_x:
        sizex = stim_res_x - ofx
    if ofy + sizey >= stim_res_y:
        sizey = stim_res_y - ofy
    return [flatIndex(x, y) for x in range(ofx, sizex+ofx) for y in range(ofy, sizey+ofy)]

def dvs2group(xs, ys, ts):
    """ Given DVS data convert it to indices and timestamps that can be used by a Brian2 Spiking
        Neuron group. 
    """
    # TODO This means no 2 distinct neurons can fire at the same us.... Needs a fix
    real_indices = [flatIndex(*xy) for xy in zip(xs, ys)]
    real_times, uniq_idxs = np.unique(ts, return_index=True)
    real_indices = np.asarray(real_indices)[uniq_idxs]
    real_times = (real_times - real_times[0])*us
    return (real_indices, real_times)

 

