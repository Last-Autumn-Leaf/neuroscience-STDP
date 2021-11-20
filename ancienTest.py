import brian2.units.allunits

from tools import *
from PlotStdpForm import PlotStdpForm

N = 1000
F = 10 * Hz
gmax = 1 / 5
# Création des neurones
tau = 10 * ms
eqs_neuron = '''
dv/dt = -v/tau : 1
'''

poisson_input = PoissonGroup(N, rates=F)

neuron = NeuronGroup(1, model=eqs_neuron, threshold='v>1', reset='v=0', method='euler')

# Création des synapses
brian2.units.allunits.radian
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
# w = clip(w + b,-gmax,0)
on_post = '''
    b += B
    w = clip(w + a,0,gmax)
'''
# PlotStdpForm(eqs_stdp,on_pre,on_post)
S = Synapses(poisson_input, neuron, model=eqs_stdp, on_pre=on_pre, on_post=on_post, method='euler')
S.connect()

S.w = 'rand() * gmax'
mon = StateMonitor(S, 'w', record=[0, 1])
s_mon = SpikeMonitor(poisson_input)

run(100 * second, report='text')

subplot(311)
plot(S.w / gmax, '.k')
ylabel('Weight / gmax')
xlabel('Synapse index')
subplot(312)
hist(S.w / gmax, 20)
xlabel('Weight / gmax')
subplot(313)
plot(mon.t / second, mon.w.T / gmax)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()
show()

