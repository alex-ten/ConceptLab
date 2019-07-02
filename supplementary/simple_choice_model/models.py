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


def simulate(X, N_trials, hits_model, alpha, beta, gamma, memcap, lpwin):
    counter = np.array([1, 1, 1, 1])
    mem = X.values.copy()[-memcap:, :]
    init_pc = np.mean(mem, axis=0)
    init_lp = np.abs(mem[:-lpwin, :].mean(axis=0) - mem[-lpwin:, :].mean(axis=0))
    init_in = np.zeros_like(init_pc)
    lps, pcs = [init_lp], [init_pc] 
    util = []
    choices, hits = np.zeros(N_trials), np.zeros(N_trials)
    
    x = np.stack([init_lp, init_pc, init_in], axis=0).T
    for t in range(N_trials):
        # Compute utility based on state x, choose the next task based on utility, 
        # and get feedback by playing the task
        U = utility(x, alpha=alpha, beta=beta, gamma=gamma)
        i = choose(softmax(U), det=False)
        counter[i] += 1
        hit = hits_model(counter[i], i+1)
        
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
        lp_vect = np.abs(mem[:-lpwin, :].mean(axis=0) - mem[-lpwin:, :].mean(axis=0))
        x[i, 0] = lp_vect[i] # LP
        
        # ========== Record data ============
        pcs.append(pc_vect)
        lps.append(lp_vect)
        util.append(U)
        choices[t] = i
        hits[t] = hit
    
    return counter/np.sum(counter), pcs, lps, choices, hits, util


def get_sub_data(sid=None, ntm=None, grp=None):
    from loc_utils import unpickle
    arrdata = unpickle('data/main_data.pkl')['main'][:, [r.ix('group'),r.ix('sid'),r.ix('trial'),r.ix('cat'),r.ix('cor'),r.ix('blkt')]]
    df_dict = {}
    for i, (col, dt) in enumerate(zip('grp,sid,trial,tid,cor,blkt'.split(','), [int,int,int,int,int,int])):
        df_dict[col] = pd.Series(arrdata[:, i], dtype=dt)
    df = pd.DataFrame(df_dict)
    
    ntm_df = lut.unpickle(ntm_path)[['sid','ntm']].groupby('sid').head(1)
    df = df.merge(ntm_df, on='sid').drop_duplicates()
    del ntm_df, arrdata
    df = df.loc[df.trial<=60]
    
    if ntm: df = df.loc[df.ntm==ntm, :]
    if grp: df = df.loc[df.grp==grp, :]
    if sid: 
        df = df.loc[df.sid==sid, :]
    else:
        sid = np.random.choice(df.sid.unique())
        df = df.loc[df.sid==sid, :]
    print('SID: {}'.format(sid))
        
    df = df.loc[:, ['blkt','tid','cor']].pivot(index='blkt', columns='tid', values='cor')
    return df