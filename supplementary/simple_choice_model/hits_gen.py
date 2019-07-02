# Python libraries
from IPython.display import display
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

# Local libraries
import vis_utils as vut
import loc_utils as lut
import ipywidgets as wid
import loc_utils as lut
from standards import *

# Local variables
main_path = 'supplementary/simple_choice_model/data/main_data.pkl' 
ntm_path = 'supplementary/simple_choice_model/data/ntm_data.pkl'
r = RAWix()
colors = ['#43799d', '#cc5b46', '#ffbb00', '#71bc78', '#43799d', '#cc5b46', '#ffbb00', '#71bc78']
tlabels = {
        1: '1D',
        2: 'I1D',
        3: '2D',
        4: 'R'}

class HitsGeneratorGUI():
    def __init__(self, bandits):
        self.grp_picker = wid.Dropdown(options=[('All', None), ('Free', 0), ('Strategic', 1)], 
                                       value=None, description='Group: ', layout=wid.Layout(width='30%'),)
        self.ntm_picker = wid.Dropdown(options=[('All', None), ('1', 1), ('2', 2), ('3', 3)], 
                                       value=None, description='NTM: ', layout=wid.Layout(width='30%'),)
        
        self.slides = {}
        self.params_dict = {}
        for tid, tsk in enumerate(bandits):
            b0slide = wid.BoundedFloatText(value=1., min=-3., max=3., step=0.01, layout=wid.Layout(width='200pt'),
                description='b0: '.format(tsk), continuous_update=False, orientation='horizontal')
            b1slide = wid.BoundedFloatText(value=1., min=-2., max=2., step=0.001, layout=wid.Layout(width='200pt'),
                description='b1: '.format(tsk), continuous_update=False, orientation='horizontal')
            label = wid.HTML(value='<b><font color="{}">{}</b>'.format(colors[tid], tsk), layout=wid.Layout(width='10pt'))
            self.slides[tid+1] = (label, b0slide, b1slide)
            self.params_dict[tid+1] = (b0slide.value, b1slide.value)
        self.sim_n = wid.IntText(value=150, description='Runs:')
        self.sim_t = wid.IntText(value=30, description='Trials/run:')
        self.reset_button = wid.Button(description='Fit to all data', button_style='warning')
        self.reset_button.on_click(lambda x: self.reset())
        
        self.fig = plt.figure('Hits generator', figsize=[8.5, 4])
        self.ax1, self.ax2 = vut.pretty(self.fig.add_subplot(1,2,1)), vut.pretty(self.fig.add_subplot(1,2,2))
        
        self.ax1.set_ylim(.4, 1.02)
        self.ax1.set_xlim(0, 250)
        self.ax2.set_ylim(.4, 1.02)
        
        self.ax1.set_title('$P(hit \mid trial)$')
        self.ax2.set_title('Simulated data')
        
        self.axes = [self.ax1, self.ax2]
        
        self.controls_on()
        
    def prob(self, trial, tid):
        b0, b1 = self.params[tid]
        return 1 / (1 + np.exp(-(b0 + b1*trial)))
    
    def generate(self, trial, tid):
        b0, b1 = self.params_dict[tid]
        prob = 1 / (1 + np.exp(-(b0 + b1*trial)))
        try:
            iter(trial)
            return (np.random.rand(trial.size) <= prob).astype(int)
        except TypeError as te:
            return int(np.random.rand() <= prob)
        
    def controls_on(self):
        _ = wid.interactive(self.fit_model, grp=self.grp_picker, ntm=self.ntm_picker)
        _dropdowns = wid.HBox([self.grp_picker, self.ntm_picker])
        _slides = wid.VBox([wid.HBox(self.slides[tid]) for tid in sorted(self.slides.keys())])
        _textboxes = wid.HBox([self.sim_n, self.sim_t, self.reset_button])
        _.update()
        
        kwargs = {'T': self.sim_t, 'N':self.sim_n}
        for k, v in self.slides.items():
            _, b0slide, b1slide = v
            kwargs['t{}b0'.format(k)] = b0slide
            kwargs['t{}b1'.format(k)] = b1slide
        __ = wid.interactive(self.plot_model, **kwargs)
        __.update()
        display(wid.VBox([_dropdowns, _slides, _textboxes]))
        
    def fit_model(self, grp, ntm):
        new_params = get_parametric(grp=grp, ntm=ntm)
        for k, v in new_params.items():
            self.params_dict[k] = v
            self.slides[k][1].value = v[0]
            self.slides[k][2].value = v[1]
        
    def plot_model(self, **kwargs):
        for ax in self.axes:
            while ax.get_lines(): ax.get_lines()[0].remove()
        for patch in self.axes[1].findobj(match=mpl.collections.PolyCollection):
            patch.remove()

        x = np.arange(0, 250)
        tasks = [1,2,3,4]
        
        params = {}
        for k, v in self.slides.items():
            b0, b1 = [kwargs['t{}b0'.format(k)], kwargs['t{}b1'.format(k)]]
            probs = 1 / (1 + np.exp(-(b0 + b1*x)))
            self.axes[0].plot(x, probs, lw=2, color=colors[k-1])
            self.params_dict[k] = (b0, b1)

            sim_data = (np.random.rand(kwargs['N'], kwargs['T']) <= probs[:kwargs['T']]).astype(int)
            csum = np.cumsum(sim_data, axis=1)
            tnum = np.arange(kwargs['T']) + 1
            mean = np.mean(csum / tnum, axis=0)
            se = sp.stats.sem(csum / tnum, axis=0)
            self.axes[1].plot(tnum-1, mean, lw=2, color=colors[k-1])
            self.axes[1].fill_between(tnum-1, mean+se, mean-se, color=colors[k-1], alpha=.3)
        self.ax2.set_xlim(.4, self.sim_t.value+1)
        self.fig.tight_layout()
        
    def reset(self):
        self.grp_picker.value, self.ntm_picker.value = None, None
        new_params = get_parametric(grp=self.grp_picker.value, ntm=self.ntm_picker.value)
        for k, v in new_params.items():
            self.params_dict[k] = v
            self.slides[k][1].value = v[0]
            self.slides[k][2].value = v[1]
        
    def view_params(self):
        for k in sorted(self.params_dict.keys()):
            print(k, self.params_dict[k])
            

def get_parametric(grp=None, ntm=None):
    arrdata = lut.unpickle(main_path)['main'][:, [r.ix('group'),r.ix('sid'),r.ix('trial'),r.ix('cat'),r.ix('cor'),r.ix('blkt')]]
    df_dict = {}
    for i, (col, dt) in enumerate(zip('grp,sid,trial,tid,cor,blkt'.split(','), [int,int,int,int,int,int])):
        df_dict[col] = pd.Series(arrdata[:, i], dtype=dt)

    df = pd.DataFrame(df_dict)
    ntm_df = lut.unpickle(ntm_path)[['sid','ntm']].groupby('sid').head(1)
    df = df.merge(ntm_df, on='sid').drop_duplicates()
    del ntm_df, arrdata

    if ntm is not None: df = df.loc[df.ntm==ntm, :]
    if grp is not None: df = df.loc[df.grp==grp, :]

    params_dict = {}
    for tid in [1,2,3,4]:
        tdf = df.loc[df.tid==tid, :]
        m = smf.logit('cor ~ blkt', data=tdf).fit(full_output=0, disp=0)
        b0, b1 = m.params['Intercept'], m.params['blkt']
        params_dict[tid] = np.array([b0, b1])
    return params_dict


def get_nonparametric(grp=None, ntm=None):
    arrdata = lut.unpickle(main_path)['main'][:, [r.ix('group'),r.ix('sid'),r.ix('trial'),r.ix('cat'),r.ix('cor'),r.ix('blkt')]]

    df_dict = {}
    for i, (col, dt) in enumerate(zip('grp,sid,trial,tid,cor,blkt'.split(','), [int,int,int,int,int,int])):
        df_dict[col] = pd.Series(arrdata[:, i], dtype=dt)

    df = pd.DataFrame(df_dict)
    ntm_df = lut.unpickle(ntm_path)[['sid','ntm']].groupby('sid').head(1)
    df = df.merge(ntm_df, on='sid').drop_duplicates()
    del ntm_df, arrdata

    if ntm: df = df.loc[df.ntm==ntm, :]
    if grp: df = df.loc[df.grp==grp, :]

    grouped = df.groupby(['tid', 'blkt'])[['cor']].mean()

    probs = np.zeros([4, 100])
    for tid in [1,2,3,4]:
        y = grouped.loc[(tid, slice(None)), :].rolling(50, min_periods=1).mean().values.squeeze()
        probs[tid-1, :] = y[:100]

    def nonparametric_model(trials, tid):
        t = trials.copy()
        t[t <= 0] = 0
        t[t >= 100] = 99
        p = probs[tid-1, t]
        return (np.random.rand(t.size) <= p).astype(int)
    
    return nonparametric_model