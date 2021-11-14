from brian2 import *
from PlotStdpForm import PlotStdpForm


def BrianSTDPExemple(F1= None,F2= None,spec_vars=None):
    N = 1000
    F = 15*Hz

    eqs_neurons = '''
    dv/dt = (ge * (Ee-v) + El - v) / taum : volt
    dge/dt = -ge / taue : 1
    taum = 10*ms  : second
    taupre = 20*ms : second
    taupost = taupre : second
    Ee = 0*mV :volt
    vt = -54*mV :volt
    vr = -60*mV :volt
    El = -74*mV :volt
    taue = 5*ms : second
    gmax = .01 :1
    '''
    gmax = .01
    input = PoissonGroup(N, rates=F)
    neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                          method='euler')




    if F1 == None :
        F1="gmax*.01 * exp(-(t_spike_b - t_spike_a)/taupost)"
    if F2 == None:
        F2="-gmax*.01*1.05 * exp((t_spike_b - t_spike_a)/taupre)"
    if spec_vars==None:
        spec_vars=""

    eqs_stdp=''' w : 1
                 t_spike_a : second 
                 t_spike_b : second
                 t0 = 0 * second : second
                 {}
                 '''.format(spec_vars)

    on_pre='''ge+=w
              t_spike_a = t
              w = clip(w + int(t_spike_b > t0) * {},0,gmax)
              t_spike_b = t0'''.format(F2)
    on_post=''' t_spike_b = t
            w = clip(w +  int(t_spike_a > t0) * {},0,gmax)
            t_spike_a = t0'''.format(F1)

    S = Synapses(input, neurons,eqs_stdp,on_pre=on_pre,on_post=on_post )

    S.connect()
    S.w = 'rand() * gmax'
    mon = StateMonitor(S, 'w', record=[0, 1])
    s_mon = SpikeMonitor(input)

    run(100*second, report='text')

    subplot(311)
    plot(S.w / gmax, '.k')
    ylabel('Weight / gmax')
    xlabel('Synapse index')
    subplot(312)
    hist(S.w / gmax, 20)
    xlabel('Weight / gmax')
    subplot(313)
    plot(mon.t/second, mon.w.T/gmax)
    xlabel('Time (s)')
    ylabel('Weight / gmax')
    tight_layout()
    show()




if __name__=='__main__' :
    #F1 post !
    #F2 Pre

    F1=" clip( B * log((t_spike_b - t_spike_a)/tau_b+ 0.1),-0.001,0 )"
    F2=" clip( A * (t_spike_b - t_spike_a)/tau_a,0,0 )"
    spec_vars=    '''
                 tau_a = 60 * ms: second
                 tau_b = 10 * ms: second
                 tau_c= 1*ms : second
                 K=0.0001 : 1
                 w_n=0.05 : 1
                z=0.7 : 1
                c=-z*w_n : 1
                phi=pi/2 : 1
                A = 0.0001 : 1
                B = A*0.25 : 1
                 '''


    courbe=False





    if courbe :
        PlotStdpForm(F1,F2,spec_vars,duration=40,title="Forme B II")
    else :
        set_device('cpp_standalone')
        BrianSTDPExemple(F1,F2,spec_vars)



