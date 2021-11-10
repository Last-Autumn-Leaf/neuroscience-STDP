from tools import *
import timeit
#Connection G to G : S

def PlotStdpForm(eqs_stdp,on_pre,on_post):
    start_scope()

    # Création des neurones
    eqs_neuron = '''
    dv/dt = -v/tau : 1
    tau : second
    '''
    G = NeuronGroup(2, model=eqs_neuron, threshold='v>1', reset='v=0', method='euler')
    G.tau = 10 * ms

    # Création des synapses
    S = Synapses(G, G, model=eqs_stdp, on_pre=on_pre, on_post=on_post, method='euler')
    S.connect(i=0, j=1)

    input_generator = SpikeGeneratorGroup(2, [], [] * ms)  # Our input layer consist of 2 neurons
    # Connectons ce générateur à nos deux neurones
    input_generator_synapses = Synapses(input_generator, G, on_pre='v_post += 2')  # Forcer des décharges
    input_generator_synapses.connect(i=[0, 1], j=[0, 1])
    # Faisons la simulation pour différents Delta t et calculons Delta w.
    deltat = np.linspace(-50, 50, num=50)
    deltaw = np.zeros(deltat.size)  # Vecteur pour les valeurs de Delta w
    # On utilise store() et restore() pour faire cette simulation, comme expliqué dans le notebook sur Brian2!
    store()
    for i in range(deltat.size):
        dt = deltat[i]
        restore()
        # On fait en sorte que les neurones déchargent à 0 ms et à |dt| ms
        # En fonction du signe de dt, les neurones vont décharger un avant l'autre
        if dt < 0:
            input_generator.set_spikes([0, 1], [-dt, 0] * ms)
        else:
            input_generator.set_spikes([0, 1], [0, dt] * ms)
        run((np.abs(dt) + 1) * ms)
        deltaw[i] = S.w[0]  # delta w est tout simplement w ici parce que w est à zéro initialement
    # Faisons le graphique de dw en fonction de dt
    plt.figure(figsize=(8, 5))
    plt.plot(deltat, deltaw, linestyle='-', marker='o')
    plt.title('STDP paramétrisée avec Brian2')
    plt.xlabel('Δt')
    plt.ylabel('Δw')
    axhline(y=0, color='black')
    # plt.ylim(min(A, B), max(A, B))
    plt.grid()
    plt.show()


def fasterPlotSTDP(eqs_stdp,on_pre,on_post,tmax=70*ms,N=100):
    start_scope()
    # Presynaptic neurons G spike at times from 0 to tmax
    # Postsynaptic neurons G spike at times from tmax to 0
    # So difference in spike times will vary from -tmax to +tmax
    G = NeuronGroup(N, 'tspike:second', threshold='t>tspike', refractory=100 * ms)
    H = NeuronGroup(N, 'tspike:second', threshold='t>tspike', refractory=100 * ms)
    G.tspike = 'i*tmax/(N-1)'
    H.tspike = '(N-1-i)*tmax/(N-1)'

    S = Synapses(G, H,
                 model=eqs_stdp, on_pre=on_pre, on_post=on_post,method='euler')
    S.connect(j='i')

    run(tmax + 1 * ms)

    plot((H.tspike - G.tspike) / ms, S.w)
    xlabel(r'$\Delta t$ (ms)')
    ylabel(r'$\Delta w$')
    axhline(0, ls='-', c='k');
    show()


if __name__=='__main__':

    eqs_stdp = '''
        w : 1
        da/dt = -a / tau_a : 1 (event-driven) 
        db/dt = -b / tau_b : 1 (event-driven)
        tau_a =20*ms:second
        tau_b =20*ms:second
        A= 0.01:1
        B= -0.01:1
    '''
    on_pre = '''
        v_post += w
        a += A
        w = w + b
    '''
    on_post = '''
        b += B
        w = w + a
    '''


    start = timeit.default_timer()
    PlotStdpForm(eqs_stdp,on_pre,on_post)
    stop = timeit.default_timer()
    print('Time Amad: ', stop - start)


    '''PlotStdpForm(eqs_stdp,on_pre,on_post)
    fasterPlstart = timeit.default_timer()
    stop = timeit.default_timer()
    print('Time brian: ', stop - start)'''