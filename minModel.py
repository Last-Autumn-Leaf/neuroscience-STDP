import numpy as np
from brian2 import units
from sklearn import datasets, model_selection
from brian2 import *
import matplotlib.pyplot as plt
# Cible de génération de code pour Brian2
prefs.codegen.target = 'cython'
set_device('cpp_standalone', build_on_run=False)

time_per_sample=350*ms
resting_time = 0.15 * units.second

N = 1000
F = 10 * Hz
gmax = 1 / 5
# Création des neurones
tau = 10 * ms
eqs_neuron = '''
dv/dt = -v/tau : 1
'''

input_group = PoissonGroup(N, rates=F)

neuron = NeuronGroup(1, model=eqs_neuron, threshold='v>1', reset='v=0', method='euler')
# Création des synapses
eqs_stdp = '''
    w : 1
    da/dt = -a / tau_a : 1 (event-driven) 
    db/dt = -b / tau_b : 1 (event-driven)

    tau_a = 20*ms : second
    tau_b = 20*ms : second
    A = 0.001 *gmax : 1
    B = -0.001 *gmax : 1
    gmax = 1./5. : 1
'''
on_pre = '''
    v_post += w
    a += A
    w = clip(w + b,0,gmax)
'''
on_post = '''
    b += B
    w = clip(w + a,0,gmax)
'''
S = Synapses(input_group, neuron, model=eqs_stdp, on_pre=on_pre, on_post=on_post, method='euler')
S.connect()
S.w = 'rand() * gmax'
mon = StateMonitor(S, 'w', record=[0, 1])
s_mon = SpikeMonitor(input_group)
net = Network(input_group, S, neuron,s_mon,mon)
array_of_rates=np.zeros((10,N))
for i in range(10):
    for j in range(N):
        array_of_rates[i,j]=i*j
number_of_epochs = 1
l=0
input_group.run_regularly('rates = array_of_rates[l]*Hz;l+=1;', dt=350*ms)
for i in range(number_of_epochs):
    print('Starting iteration %i' % i)
    for  i in range(10):
        # Configurer le taux d'entrée
        #input_group.rates = i*10 * units.Hz
        # Simuler le réseau
        net.run(time_per_sample)
        a=s_mon.count
        # Laisser les variables retourner à leurs valeurs de repos
        net.run(resting_time)
        device.build(directory='output', compile=True, run=True, debug=False)
        print(S.w)

