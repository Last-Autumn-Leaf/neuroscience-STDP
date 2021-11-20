import numpy as np
from brian2 import units
from sklearn import datasets, model_selection
from brian2 import *
import matplotlib.pyplot as plt
# Cible de génération de code pour Brian2
prefs.codegen.target = 'cython'
#set_device('cpp_standalone', build_on_run=False)
import time
import os


duration = 1  # seconds
freq = 440  # Hz

print('importation...')
X_all, y_all = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, data_home=None)

subset=3000
test_size=int(subset/5)
print('train size :',subset-test_size)
print('test size :',test_size)
X = X_all[:subset]
y = y_all[:subset]
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=test_size)

# Fixons le seed aléatoire afin de pouvoir reproduire les résultats
np.random.seed(0)
# Horloge de Brian2
defaultclock.dt = 0.5 * units.ms


time_per_sample =   0.35 * units.second
resting_time = 0.15 * units.second

v_rest_e = -65. * units.mV
v_rest_i = -60. * units.mV

v_reset_e = -65. * units.mV
v_reset_i = -45. * units.mV

v_thresh_e = -52. * units.mV
v_thresh_i = -40. * units.mV

refrac_e = 5. * units.ms
refrac_i = 2. * units.ms

tc_theta = 1e7 * units.ms
theta_plus_e = 0.05 * units.mV

tc_pre_ee = 20 * units.ms
tc_post_1_ee = 20 * units.ms
tc_post_2_ee = 40 * units.ms

# Taux d'apprentissage
nu_ee_pre =  0.0001
nu_ee_post = 0.01

Ne=400
Ni=Ne

wmax_ee = 1.0

delay={}
delay['ee_input'] = (0*ms,10*ms)
delay['ei_input'] = (0*ms,5*ms)
input_intensity = 2.

input_group = PoissonGroup(X_train.shape[1],0*Hz)

neuron_model = '''
    dv/dt = ((v_rest_e - v) + (I_synE + I_synI) / nS) / tau  : volt (unless refractory)

    I_synE =  ge * nS * -v           : amp

    I_synI =  gi * nS * (d_I_synI-v) : amp

    dge/dt = -ge/(1.0*ms)            : 1

    dgi/dt = -gi/(2.0*ms)            : 1

    tau                              : second (constant, shared)

    d_I_synI                         : volt (constant, shared)

    dtheta/dt = -theta / (tc_theta)  : volt
'''

excitatory_group = NeuronGroup(
    N=Ne, model=neuron_model, refractory=refrac_e,
    threshold='v>v_thresh_e+ theta - 20.0*mV', reset='v=v_rest_e; theta += theta_plus_e', method='euler')
excitatory_group.tau = 100 * units.ms
excitatory_group.d_I_synI = -100. * units.mV

inhibitory_group = NeuronGroup(
    N=Ni, model=neuron_model, refractory=refrac_i,
    threshold='v>v_thresh_i', reset='v=v_rest_i', method='euler')
inhibitory_group.tau = 10 * units.ms
inhibitory_group.d_I_synI = -85. * mV

synapse_model = "w : 1"

stdp_synapse_model = '''
    w : 1

    plastic : boolean (shared) # Activer/désactiver la plasticité

    post2before : 1

    dpre/dt   =   -pre/(tc_pre_ee) : 1 (event-driven)

    dpost1/dt  =  -post1/(tc_post_1_ee) : 1 (event-driven)

    dpost2/dt  =  -post2/(tc_post_2_ee) : 1 (event-driven)
'''

stdp_pre = '''
    ge_post += w

    pre = 1.

    w = clip(w - nu_ee_pre * post1*plastic, 0, wmax_ee)
'''

stdp_post = '''
    post2before = post2

    w = clip(w + nu_ee_post * pre * post2before*plastic, 0, wmax_ee)

    post1 = 1.

    post2 = 1.
'''

input_synapse = Synapses(input_group, excitatory_group, model=stdp_synapse_model, on_pre=stdp_pre, on_post=stdp_post)
input_synapse.connect(True)  # Fully connected
deltaDelay = delay['ee_input'][0] - delay['ee_input'][1]
input_synapse.delay = 'deltaDelay*rand() '
input_synapse.plastic = True
input_synapse.w = '(rand()+0.1)*0.3'

# e_i_synapse = Synapses(excitatory_group, inhibitory_group, model=stdp_synapse_model, on_pre=stdp_pre, on_post=stdp_post)
e_i_synapse = Synapses(excitatory_group, inhibitory_group, model=synapse_model, on_pre="ge_post += w")
e_i_synapse.connect(True, p=0.0025)
e_i_synapse.w = 'rand()*10.4'

i_e_synapse = Synapses(inhibitory_group, excitatory_group, model=synapse_model, on_pre="gi_post += w")
i_e_synapse.connect(True, p=0.9)
i_e_synapse.w = 'rand()*17.0'

e_monitor = SpikeMonitor(excitatory_group, record=False)
net = Network(input_group, excitatory_group, inhibitory_group,
              input_synapse, e_i_synapse, i_e_synapse, e_monitor)

# -------- ENTRRAINEMENT -----------------

spikes = np.zeros((10, len(excitatory_group)))
old_spike_counts = np.zeros(len(excitatory_group))

# Entrainement
number_of_epochs = 1
mean_of_w=[]
start = time.time()
for i in range(number_of_epochs):

    print('Starting iteration %i' % i)
    for j, (sample, label) in enumerate(zip(X_train.values, y_train)):

        # Afficher régulièrement l'état d'avancement
        if (j % 10) == 0:
            print("Running sample %i out of %i" % (j, len(X_train)))

        # Normaliser les poids
        weight_matrix = np.zeros((784, 400))
        weight_matrix[input_synapse.i, input_synapse.j] = input_synapse.w
        colSums = np.sum(weight_matrix, axis=0)
        colFactors = 78 / colSums
        for k in range(Ne):
            weight_matrix[:, k] *= colFactors[k]
        input_synapse.w = weight_matrix[input_synapse.i, input_synapse.j]
        mean_of_w.append(np.mean(input_synapse.w))

        # Configurer le taux d'entrée
        input_group.rates = sample / 8. * input_intensity * units.Hz
        # Simuler le réseau
        net.run(time_per_sample)


        # Enregistrer les décharges
        spikes[int(label)] += e_monitor.count - old_spike_counts
        # Gardons une copie du décompte de décharges pour pouvoir calculer le prochain
        old_spike_counts = np.copy(e_monitor.count)

        # Arrêter l'entrée
        input_group.rates = 0 * units.Hz
        # Laisser les variables retourner à leurs valeurs de repos
        net.run(resting_time)




end = time.time()
print('----time training: ',end - start)
# --------- TEST

labeled_neurons = np.argmax(spikes, axis=1)
print(labeled_neurons)

# Déasctiver la plasticité STDP
input_synapse.plastic = False

num_correct_output = 0
start = time.time()
for i, (sample, label) in enumerate(zip(X_test.values, y_test)):
    # Afficher régulièrement l'état d'avancement
    if (i % 10) == 0:
        print("Running sample %i out of %i" % (i, len(X_test)))

    # Configurer le taux d'entrée
    # ATTENTION, vous pouvez utiliser un autre type d'encodage
    input_group.rates = sample / 8. * input_intensity * units.Hz

    # Simuler le réseau
    net.run(time_per_sample)

    # Calculer le nombre de décharges pour l'échantillon
    current_spike_count = e_monitor.count - old_spike_counts
    # Gardons une copie du décompte de décharges pour pouvoir calculer le prochain
    old_spike_counts = np.copy(e_monitor.count)

    # Prédire la classe de l'échantillon
    output_label = np.argmax(current_spike_count)

    # Si la prédiction est correcte
    if output_label == int(label):
        num_correct_output += 1

    # Laisser les variables retourner à leurs valeurs de repos
    net.run(resting_time)
end = time.time()
print('----time test: ',end - start)
print("The model accuracy is : %.3f" % (num_correct_output / len(X_test)))
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
os.system('say "Le programme est termine, precision de %.3f"'% (num_correct_output / len(X_test)))
hist(input_synapse.w , 20)
show()
