from tools import *
import timeit


def PlotStdpForm(F1= None,F2= None,specials_var= None,duration=50,title=None):

    if F1 == None :
        F1="gmax*.01 * exp(-(t_spike_b - t_spike_a)/taupost)"
    if F2 == None:
        F2="-gmax*.01*1.05 * exp((t_spike_b - t_spike_a)/taupre)"
    if specials_var==None:
        specials_var = """
                taupre=20*ms :second
                taupost=20*ms:second
                gmax=0.01 :1
            """

    tau = 10 * ms
    eqs_neurons = '''
    dv/dt = -v/tau : 1
    '''
    G = NeuronGroup(2, model=eqs_neurons, threshold='v>1', reset='v=0', method='euler')


    t0 = 0 * second

    eqs_stdp = '''
        w : 1
        t_spike_a : second 
        t_spike_b : second
        {} 
    '''.format(specials_var)

    on_pre = '''
        t_spike_a = t
        w = w + int(t_spike_b > t0) * {}     
        t_spike_b = t0
    '''.format(F2)
    on_post = '''
        t_spike_b = t
        w = w +  int(t_spike_a > t0) * {}   
        t_spike_a = t0
    '''.format(F1)

    S = Synapses(G, G, model=eqs_stdp, on_pre=on_pre, on_post=on_post, method='euler')
    S.connect(i=0, j=1)

    input_generator = SpikeGeneratorGroup(2, [], [] * ms)
    input_generator_synapses = Synapses(input_generator, G, on_pre='v_post += 2')
    input_generator_synapses.connect(i=[0, 1], j=[0, 1])

    # Faisons la simulation pour différents Delta t et calculons Delta w.
    deltat = np.linspace(-duration, duration, num=duration)
    deltaw = np.zeros(deltat.size)  # Vecteur pour les valeurs de Delta w

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


    plt.figure(figsize=(8, 5))
    plt.plot(deltat, deltaw, linestyle='-', marker='o')
    if title ==None:
         title = 'STDP paramétrisée avec Brian2'
    plt.title(title)
    plt.xlabel('Δt')
    plt.ylabel('Δw')
    axhline(y=0, color='black')
    plt.grid()
    plt.show()

if __name__=='__main__':

    start_scope()

    F1 = "gmax*.01 * exp(-(t_spike_b - t_spike_a)/taupost)"
    F2 = "-gmax*.01*1.05 * exp((t_spike_b - t_spike_a)/taupre)"
    specials_var="""
        taupre=20*ms :second
        taupost=20*ms:second
        gmax=0.01 :1
    """
    PlotStdpForm(F1,F2,specials_var)
