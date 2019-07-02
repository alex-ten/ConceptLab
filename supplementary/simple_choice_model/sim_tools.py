# Python libraries
from IPython.display import display
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import warnings
import contextlib

# Local libraries
import vis_utils as vut
import loc_utils as lut
import ipywidgets as wid
import loc_utils as lut
from standards import *

# Global variables
main_path = 'supplementary/simple_choice_model/data/main_data.pkl' 
ntm_path = 'supplementary/simple_choice_model/data/ntm_data.pkl'
r = RAWix()
colors = ['#43799d', '#cc5b46', '#ffbb00', '#71bc78', '#43799d', '#cc5b46', '#ffbb00', '#71bc78']
tlabels = {
        1: '1D',
        2: 'I1D',
        3: '2D',
        4: 'R'}


def dummy(n, i):
    dummy = np.zeros(n)
    dummy[i] = 1
    return dummy


def utility(x, alpha=1, beta=1, gamma=0):
    ''' Utility function
    x must be a row (or an array of rows) 
    containing 3 values: LP, PC, and I '''
    return np.sum(x * np.array([alpha, beta, gamma]), axis=1)


def softmax(U):
    return np.exp(U) / np.sum(np.exp(U))


def choose(a, det=False):
    ''' Choice function
    if det  : choose an option with maximum utility
    if !det : choose an option probabilistically, in proportion to utility'''
    if det:
        return np.argmax(a)
    else:
        cum_probs = np.cumsum(a, axis=0)
        rand = np.random.rand(a.shape[-1])
        return (rand<cum_probs).argmax()

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
    
pd.options.display.float_format = '{:.5f}'.format
# interactive_util = wid.interact(plot_sim, X=wid.fixed(X), t=trial_,
#                                 alpha=alpha_, beta=beta_, gamma=gamma_, 
#                                 ax1=wid.fixed(ax1), ax2=wid.fixed(ax2), ax3=wid.fixed(ax3),
#                                 seed=wid.fixed(2))

def get_sub_data(sid=None, grp=None, ntm=None):
    from loc_utils import unpickle
    arrdata = unpickle(main_path)['main'][:, [r.ix('group'),r.ix('sid'),r.ix('trial'),r.ix('cat'),r.ix('cor'),r.ix('blkt')]]
    df_dict = {}
    for i, (col, dt) in enumerate(zip('grp,sid,trial,tid,cor,blkt'.split(','), [int,int,int,int,int,int])):
        df_dict[col] = pd.Series(arrdata[:, i], dtype=dt)
    df = pd.DataFrame(df_dict)

    ntm_df = lut.unpickle(ntm_path)[['sid','ntm']].groupby('sid').head(1)
    df = df.merge(ntm_df, on='sid').drop_duplicates().astype(int)
    del ntm_df, arrdata
    df = df.loc[df.trial<=60]
    if ntm: df = df.loc[df.ntm==ntm, :]
    if grp: df = df.loc[df.grp==grp, :]
    if sid: 
        df = df.loc[df.sid==int(sid), :]
    else:
        sid = np.random.choice(df.sid.unique())
        df = df.loc[df.sid==sid, :]

    df = df.loc[:, ['blkt','tid','cor']].pivot(index='blkt', columns='tid', values='cor')
    return df, sid


class Simulator():
    def __init__(self, nb_trials, hits_generator, live=False, seed=1):
        self._first = True
        self.nb_trials = nb_trials
        self.seed = seed
        self.memcap = 15
        self.lpwin = 1
        
        # WIDGETS
        self.alpha = wid.FloatSlider(min=-10, max=10, value=1, description='alpha', continuous_update=live, layout=wid.Layout(width='80%'))
        self.beta = wid.FloatSlider(min=-10, max=10, value=1, description='beta', continuous_update=live, layout=wid.Layout(width='80%'))
        self.gamma = wid.FloatSlider(min=-10, max=10, value=1, description='gamma', continuous_update=live, layout=wid.Layout(width='80%'))
        self.trial = wid.IntSlider(min=0, max=nb_trials-1, value=nb_trials-1, description='trial', continuous_update=live, layout=wid.Layout(width='80%'))
        self.grp_picker = wid.Dropdown(options=[('All', None), ('Free', 0), ('Strategic', 1)], 
                                       value=None, description='Group: ', layout=wid.Layout(width='20%'))
        self.ntm_picker = wid.Dropdown(options=[('All', None), ('1', 1), ('2', 2), ('3', 3)], 
                                       value=None, description='NTM: ', layout=wid.Layout(width='20%'))
        self.sid_picker = wid.Text(value='', placeholder='ID or blank', description='Subject ID', layout=wid.Layout(width='20%'))
        self.update_button = wid.Button(description='Update initial state', button_style='info')
        self.update_button.on_click(lambda x: self.update_init_state())
        
        self.sim_button = wid.Button(description='Simulate', button_style='success')
        self.sim_button.on_click(lambda x: self.run_sim())
        
        self.hits_generator = hits_generator
        
        self.fig = plt.figure('Simulation', figsize=[8,6])

        self.ax1 = vut.pretty(self.fig.add_subplot(221))
        self.ax1.set_ylim(-.5, 3.5)
        self.ax1.set_yticks([0,1,2,3])
        self.ax1.set_yticklabels([tlabels[i] for i in [4,3,2,1]])
        self.ax1.set_xticks(list(range(15)))
        self.ax1.set_xticklabels(list(range(15)))
        self.ax1.imshow(np.zeros([4,15]), cmap='binary')
        
        self.ax2 = vut.pretty(self.fig.add_subplot(222))
        self.ax2.set_xticks([0,1,2,3])
        self.ax2.set_xticklabels([tlabels[i] for i in [1,2,3,4]]) 
        
        self.ax3 = vut.pretty(self.fig.add_subplot(212))
        self.ax3.set_ylim(-4, 1)
        self.ax3.set_yticks([-3,-2,-1,0])
        self.ax3.set_yticklabels([tlabels[i] for i in [4,3,2,1]])
        
        self.init_state=None
        self.controls_on()
        
    def controls_on(self):
        display(wid.HBox([self.sid_picker, self.grp_picker, self.ntm_picker]), 
                self.update_button, self.sim_button)
        display(wid.VBox([self.alpha, self.beta, self.gamma, self.trial]))

    def update_init_state(self):
        sid, grp, ntm = self.sid_picker.value, self.grp_picker.value, self.ntm_picker.value
        if sid=='': sid=None
        new_init_state, new_init_state_id = get_sub_data(sid, grp, ntm)
        if new_init_state.empty:
            print('Subject {} not found in (GRP=\'{}\' & NTM=\'{}\'). Init state not updated, try another search'.format(sid, grp, ntm))
        else: 
            self.init_state, self.init_state_id = new_init_state, new_init_state_id
            self.ax1.imshow(np.flipud(self.init_state.T.values), cmap='binary')
            self.sid_picker.placeholder = str(self.init_state_id)
            self.sid_picker.value = ''
            self.ax1.set_title('SID: {}'.format(self.init_state_id))
        
    def simulate(self, alpha, beta, gamma):
        counter = np.array([1, 1, 1, 1])
        mem = self.init_state.values.copy()[-self.memcap:, :]
        init_pc = np.mean(mem, axis=0)
        init_lp = np.abs(mem[:-self.lpwin, :].mean(axis=0) - mem[-self.lpwin:, :].mean(axis=0))
        init_in = np.zeros_like(init_pc)
        lps, pcs = [init_lp], [init_pc] 
        util = []
        choices, hits = np.zeros(self.nb_trials), np.zeros(self.nb_trials)

        x = np.stack([init_lp, init_pc, init_in], axis=0).T
        for t in range(self.nb_trials):
            # Compute utility based on state x, choose the next task based on utility, 
            # and get feedback by playing the task
            U = utility(x, alpha=alpha, beta=beta, gamma=gamma)
            i = choose(softmax(U), det=False)
            counter[i] += 1
            hit = self.hits_generator.generate(counter[i], i+1)

            # ========== Update memory ==========
            # 1. Update last choice
            x[:, 2] = 0 
            x[i, 2] = 1

            # 2. Update hits memory
            mem[:-1, :] = mem[1:, :]
            mem[-1, i] = hit

            # 3. Update expected reward (PC)
            pc_vect = np.mean(mem, axis=0)
            x[i, 1] = pc_vect[i] # PC

            # 4. Update LP
            lp_vect = np.abs(mem[:-self.lpwin, :].mean(axis=0) - mem[-self.lpwin:, :].mean(axis=0))
            x[i, 0] = lp_vect[i] # LP

            # ========== Record data ============
            pcs.append(pc_vect)
            lps.append(lp_vect)
            util.append(U)
            choices[t] = i
            hits[t] = hit
        return counter/np.sum(counter), pcs, lps, choices, hits, util
    
    def plot_sim(self, t, alpha, beta, gamma):
        with temp_seed(self.seed):
            tot, pc, lp, choices, hits, util = self.simulate(alpha, beta, gamma)
            while len(self.ax2.findobj(match=mpl.patches.Rectangle)) > 1: 
                self.ax2.findobj(match=mpl.patches.Rectangle)[0].remove()
            rects = self.ax2.bar(np.arange(self.init_state.shape[1]), tot, color='white', edgecolor='k')

            while self.ax3.get_lines():
                self.ax3.get_lines()[0].remove()
            self.ax3.axvline(t, color='k')

            inds = np.arange(choices.size)
            hmask = hits.astype(bool)
            self.ax3.plot(inds[hmask], -choices[hmask], c='green', marker='|', ls='', ms=12, mew=2)
            self.ax3.plot(inds[~hmask], -choices[~hmask], c='red', marker='|', ls='', ms=12, mew=2)

            choices_binary = np.zeros([choices.size, 4])
            choices_binary[np.arange(choices.size), choices.astype(int)] = 1
            NA = ['N/A','N/A','N/A','N/A']
            nohit = [False, False, False, False]
            out = pd.DataFrame({
                'tid': [tlabels[tid] for tid in [1,2,3,4]], 
                'LP(t-)': lp[t], 
                'PC(t-)': pc[t], 
                'I(t-)': choices_binary[t-1, :].astype(int) if t else np.zeros(K).astype(int),
                'utility(t)': util[t],
                'p(t)': softmax(util[t]),
                'choice(t)': choices_binary[t, :].astype(bool),
                'hit(t)': choices_binary[t, :].astype(bool) if hits[t] else nohit,
                'LP(t+)': lp[t+1], 
                'PC(t+)': pc[t+1], 
            }).set_index('tid')
            display(out)
            return out
            
    def run_sim(self):
        if self._first:
            self._first = False
            self.sim = wid.interactive(self.plot_sim,
                t = self.trial,
                alpha = self.alpha, 
                beta = self.beta, 
                gamma = self.gamma)
            self.sim.update()
        else:
            self.sim.update()       