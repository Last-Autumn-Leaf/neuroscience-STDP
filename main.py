import matplotlib.pyplot as plt

from tools import *
start_scope()
duree=100
tau = 10 * ms
eqs_neuron = '''
dv/dt = -v/tau : 1
'''
G = NeuronGroup(2, model=eqs_neuron, threshold='v>1', reset='v=0', method='euler')
tau_a = tau_b = 20 * ms

tau_c= 1*ms
K=0.01
w_n=0.05
z=0.7
c=-z*w_n
phi=np.pi/2

A = 0.01
B = -A

# Cette variable nous permet de réinitialiser les instants de décharge après que la STDP s'opère dans la synapse
# On va utiliser la condition int(t_spike_a > t0) pour évaluer si oui ou non on opère le changement de poids
t0 = 0 * second

eqs_stdp = '''
    w : 1
    t_spike_a : second 
    t_spike_b : second
'''
# On peut avoir accès au temps avec la variable t dans la syntaxe des équations de Brian2
on_pre = '''
    v_post += w
    t_spike_a = t
    w = w + int(t_spike_b > t0) * B * exp((t_spike_b - t_spike_a)/tau_b)      # le cas Delta t < 0
    t_spike_b = t0
'''
on_post = '''
    t_spike_b = t
    w = w +  int(t_spike_a > t0) * K*exp(c* (t_spike_b - t_spike_a)/tau_c )*sin( (w_n*(t_spike_b - t_spike_a)/tau_c) +phi)
    t_spike_a = t0
'''



S = Synapses(G, G, model=eqs_stdp, on_pre=on_pre, on_post=on_post, method='euler')

# Création d'une connexion synaptique
S.connect(i=0, j=1)

# Générons maintenant des entrées pour nos neurones
input_generator = SpikeGeneratorGroup(2, [], [] * ms)  # Our input layer consist of 2 neurons
# Connectons ce générateur à nos deux neurones
input_generator_synapses = Synapses(input_generator, G, on_pre='v_post += 2')  # Forcer des décharges
input_generator_synapses.connect(i=[0, 1], j=[0, 1])

# Faisons la simulation pour différents Delta t et calculons Delta w.
deltat = np.linspace(-duree, duree, num=duree)
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

# Faisons le graphique de dw en fonction de dt
plt.figure(figsize=(8, 5))
plt.plot(deltat, deltaw, linestyle='-', marker='o')
plt.title('STDP paramétrisée avec Brian2')
plt.xlabel('Δt')
plt.ylabel('Δw')
axhline(y=0, color='black')
#plt.ylim(min(A, B), max(A, B))
plt.grid()
plt.show()



#G*exp(c* (t_spike_b - t_spike_a)/tau_c )*sin( (w_n*(t_spike_b - t_spike_a)/tau_c) +phi)