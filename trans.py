from support import *

##   SOMEWHAT CONSTANTS
defaultclock.dt = 1*us
tau = 10*ms
eqs = '''
dv/dt = (I - v) / tau : 1 (unless refractory)  # Pick any model
I = 0: 1
'''

def main():
    file_name = '../data/M1a_expand.aedat'
    bytes2read = 30000
    # Finally load and process the data once
    rdata = loadaerdat(datafile=file_name, length=bytes2read, version='aedat', debug=0, camera='DVS128')
    #rdata = dvs2group(xs, ys, ts)
    field_sizes = [16, 32]
    strides = [16, 8, 4]

    for field_size in field_sizes:
        surf_names, surfs = create_surfaces(field_size)

        for stride in strides:
            if stride > field_size:
                continue
            for s_num in range(len(surfs)):
                print(field_size, stride, surf_names[s_num])
                surface = surfs[s_num]
                #print(surf.shape)
                name = '{}_{}_{}field_{}stride.png'.format(file_name, surf_names[s_num], field_size, stride)
                voltMon, spikeMon_opt, locs, rcsizes = make_network(surface, rdata, field_size, stride)           # Make network
                graph_network(voltMon, spikeMon_opt, locs, rcsizes, surface, rdata, field_size, save=name)                    # Graph network

def create_surfaces(field_size):
    dist = lambda p1, p2: np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    field_sqrd = field_size * field_size
    surf_names = 'CERLDU'
    surfs = []
    ##                              Add expansions
    mid = (field_size - 1) / 2 
    max_dist = dist((0,0), (mid, mid))
    surfs.append(np.array([ dist((i%field_size, i // field_size), (mid, mid)) for i in range(field_sqrd)])/max_dist) # Contraction
    surfs.append(1 - surfs[-1])                                                                   # Expansion
    ##                              Add horizontal and vertical
    surfs.append(np.array([(i%field_size)/field_size for i in range(field_sqrd)]))                #left active
    surfs.append(1 - surfs[-1])
    surfs.append(np.array([(i//field_size)/field_size for i in range(field_sqrd)]))         #down active
    surfs.append(1 - surfs[-1])
    return (surf_names, surfs)

def make_network(surface, rdata, field_size=32, stride=8):
    start_scope()
    real_indices, real_times = dvs2group(rdata[1], rdata[2], rdata[0])
    
    # build receptive fields
    field_sqrd = field_size * field_size
    inp = SpikeGeneratorGroup(RES, real_indices, real_times)

    G, S, locs = conv_layer(inp, (RESY, RESY), surface, (field_size, field_size), stride)
    rcsizes = [(field_size, field_size) for i in range(len(locs))]
    G.v = 0.1

    voltMon = StateMonitor(G, 'v', record=True)
    spikeMon_opt = SpikeMonitor(G)
    
    run(60*ms)

    return (voltMon, spikeMon_opt, locs, rcsizes)


def graph_network(voltMon, spikeMon_opt, locs, rcsizes, surface, rdata, field_size, save=False):
    f = figure(figsize=(14, 12))
    suptitle('{} #neurons: '.format(save) + str(len(locs)))
    subplot(221)
    for n in range(len(locs)):
        plot(voltMon.t/ms, voltMon.v[n], '-'+COLOURS[n%len(COLOURS)], lw=1, label='N'+str(n))

    axhline(0.7, ls=':', c='g', lw=3)  # Threshold line
    xlabel('Time (ms)')
    ylabel('v')
    title('Membrane potentials')

    #legend(loc='best')

    # Plot image and RC locations
    ax = subplot(222)
    plot(rdata[1], rdata[2], '.k')
    for n in range(len(locs)):
        ax.add_patch(Rectangle(locs[n], *(rcsizes[n]), lw=3, color=COLOURS[n%len(COLOURS)], fill=False))
    title('Receptive fields of each kernel')

    subplot(223)
    vis_surface(surface, field_size, field_size)
    title('Convolution kernel')

    subplot(224)
    title('Number of output spikes from each neuron in second layer')
    vis_output(spikeMon_opt, np.sqrt(len(locs)), np.sqrt(len(locs)))
    
    if save:
        print("Saving:", save)
        savefig(save, bbox_inches='tight')

def vis_output(spikeMon, resx, resy):
    """ Given a spikeMonitor with variables t and i identifying neurons convert
        back to 2D coordinates and plot the response
    """
    times = spikeMon.t
    indexs_2dx = [i % resx for i in spikeMon.i]
    indexs_2dy = [int(i / resy) for i in spikeMon.i]
    
    heatmap, xedges, yedges = np.histogram2d(indexs_2dx, indexs_2dy, bins=resx)
    extent = [0,resx, 0,resy]#[xedges[0], xedges[-1], yedges[0], yedges[-1]]

    imshow(heatmap.T, extent=extent, origin='lower', interpolation='None')
    colorbar()

def vis_surface(surface, sizex=None, sizey=None):
    """ Given a surface draw it as a heatmap. If size not supplied, render in current
        shape, otherwise reshape to sizes given. Both arguements required or
        no reshaping will happen.
    """
    res = surface
    if sizex != None and sizey != None:
        res = np.zeros(shape=(sizey, sizex))
        for i in range(sizey):
            res[i, :] = surface[i*sizex : (i+1)*sizex]
            
    imshow(res, interpolation='None')
    colorbar()
    
def conv_layer(layer, layer_rc, kernel, kernel_rc, stride):
    """ Given a neuron layer and a surface (kernel of weights), create a set of synapses
        and a new layer that convolves that surface with the neuron layer with stride stride. 
        Returns tuple of (new_layer, new_synapses, locations)

        layer - a (NeuronGroup) group of neurons being input to new layer
        layer_rc - number of rows and colums of layer as tuple (row, col)
        kernel - 1D array of weights to be used as convolution kernel
        kernel_rc - rows and cols of kernel
        stride - stride between kernel convolutions
        locations - location of each receptive field
    """
    kr, kc = kernel_rc
    lr, lc = layer_rc
    
    r, c = (0, 0)
    locs = []  # Compute bottom left corner of each position for kernel
    while c + kc <= lc:    # TODO this can be faster if done analytically... 
        r = 0
        while r + kr <= lr:
            locs.append((r,c))
            r += stride
        c += stride
            
    print("Loc Num: ", len(locs))        
    ng = NeuronGroup(len(locs), eqs, threshold='v>0.7', reset='v=0.1', method='linear', refractory=5*ms)
    syns = Synapses(layer, ng, 'w : 1', on_pre='v_post += w')
    
    for n in range(len(locs)):
        arr = np.array(rec_field(*(locs[n]), kr, kc))
        syns.connect(i=arr, j=n)
        syns.w[:, n] = kernel*0.05
    
    return (ng, syns, locs)
    
if __name__ == '__main__':
    main()
