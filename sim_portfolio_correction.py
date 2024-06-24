# like sim_fake_news, but slight improvement to step1_backward
# and then adding additional code to compute the Jacobian correction

import numpy as np
import sim_steady_state_fast as sim
from numba import njit

def ss_add_lotteries(ss):
    """Add lotteries and D_next to SS"""
    ss_upd = ss.copy()
    a_i_ss, a_pi_ss = sim.interpolate_lottery_loop(ss.internals['hh']['a'], ss.internals['hh']['a_grid'])
    ss_upd.internals['hh']['a_i'] = a_i_ss
    ss_upd.internals['hh']['a_pi'] = a_pi_ss
    ss_upd.internals['hh']['D_next'] = sim.forward_policy(ss.internals['hh']['D'], a_i_ss, a_pi_ss)
    return ss_upd

def backward_iteration(Va, Pi, a_grid, y, r, beta, eis):
    # step 1: discounting and expectations
    Wa = beta * Pi @ Va
    
    # step 2: solving for asset policy using the first-order condition
    c_endog = Wa**(-eis)
    coh = y[:, np.newaxis] + (1+r)*a_grid
    # SPEEDUP: interpolation exploits monotonicity and avoids pure-Python for loop
    a = sim.interpolate_monotonic_loop(coh, c_endog + a_grid, a_grid)
        
    # step 3: enforcing the borrowing constraint and backing out consumption
    # SPEEDUP: enforce constraint in place, without creating new arrays, stop when unnecessary
    sim.setmin(a, a_grid[0])
    c = coh - a
    
    # step 4: using the envelope condition to recover the derivative of the value function
    Va = (1+r) * c**(-1/eis)
    
    return Va, a, c


def jacobian_with_correction(ss, shocks, T, no_con=False):
    """Gives Jacobian of A and C at horizon 'T' of standard incomplete markets
    model around steady state 'ss', with respect to each input shock in 'shocks'.
    'shocks' is a dict with entries (i, shock), where i is the arbitrary
    name given to a shock, and 'shock' is itself a dict with entries
    (k, dx) that specify by how much 'dx' shock perturbs each input 'k'."""
    # note: now modified to account for effects of portfolios!

    # step 1 for all shocks i, allocate to curlyY[o][i] and curlyD[i]
    curlyY = {'A': {}, 'C': {}}
    curlyD, curlyD_corr, curlyT, curlyWa, curlylambda = {}, {}, {}, {}, {}
    for i, shock in shocks.items():
        (curlyYi, curlyD[i], curlyD_corr[i],
            curlyT[i], curlyWa[i], curlylambda[i]) = step1_backward(ss, shock, T, 1E-4, no_con)
        curlyY['A'][i], curlyY['C'][i] = curlyYi['A'], curlyYi['C']
    
    # step 2 for all outputs o of interest (here A and C)
    curlyE = {}
    for o in ('A', 'C'):
        curlyE[o] = sim.expectation_functions(ss.internals['hh'][o.lower()], ss.internals['hh']['Pi'], ss.internals['hh']['a_i'], ss.internals['hh']['a_pi'], T)
                                            
    # steps 3 and 4: build fake news matrices, convert to Jacobians
    Js = {'A': {}, 'C': {}}
    Js_corr = {'A': {}, 'C': {}}
    for o in Js:
        for i in shocks:
            F = np.empty((T, T))
            F[0, :] = curlyY[o][i]
            F[1:, :] = curlyE[o][:T-1].reshape(T-1, -1) @ curlyD[i].reshape(T, -1).T
            Js[o][i] = J_from_F(F)
            Js_corr[o][i] = curlyE[o].reshape(T, -1) @ curlyD_corr[i].reshape(T, -1).T
    
    return Js, Js_corr, curlyT, curlyWa, curlylambda


def step1_backward(ss, shock, T, h=1E-4, no_con=False):
    """Performs step 1 of fake news algorithm, finding curlyY and curlyD up to
    horizon T given 'shock', which is a dict mapping inputs 'k' to how much they
    are shocked by. Use one-sided numerical diff, scaling down shock by 'h'."""
    # NOTE: now obtaining Jacobian correction curlyD_corr as well!
    # see "simple complete market correction.pdf" for details

    ss_inputs = {k: ss.internals['hh'][k] for k in ('Va', 'Pi', 'a_grid', 'y')}
    ss_inputs_agg = {k: ss[k] for k in ('r','beta', 'eis')}
    ss_inputs = {**ss_inputs, **ss_inputs_agg}

    R, Waa, Wa, upp, Dbeg, apol_diff, agrid_diff_aug = prelim_ss(ss)
    sensitivity = Wa / (R*Waa)
    if no_con:
        sensitivity[:, 0] = 0 # if you're at borr constraint, you can't be active in markets
    Lambda = -np.vdot(Dbeg, sensitivity)

    # allocate space for results
    curlyY = {'A': np.empty(T), 'C': np.empty(T)}
    curlyD = np.empty((T,) + ss.internals['hh']['D'].shape)
    curlyD_corr = np.empty_like(curlyD)
    curlyWa = np.empty_like(curlyD)
    curlyT = np.empty_like(curlyD)
    curlylambda = np.empty(T)
    
    # backward iterate
    for s in range(T):
        if s == 0:
            # at horizon of s=0, 'shock' actually hits, override ss_inputs with shock
            shocked_inputs = {k: ss_inputs[k] + h*shock[k] for k in shock}
            Va, a, c = backward_iteration(**{**ss_inputs, **shocked_inputs})
        else:
            # now the only effect is anticipation, so it's just Va being different
            Va, a, c = backward_iteration(**{**ss_inputs, 'Va': Va})
        
        # aggregate effects on A and C
        curlyY['A'][s] = np.vdot(ss.internals['hh']['D'], a - ss.internals['hh']['a']) / h
        curlyY['C'][s] = np.vdot(ss.internals['hh']['D'], c - ss.internals['hh']['c']) / h
        
        # what is effect on one-period-ahead distribution?
        da_pi = (a - ss.internals['hh']['a']) / apol_diff / h
        curlyD[s] = ss.internals['hh']['Pi'].T @ forward_policy_shock(ss.internals['hh']['D'], ss.internals['hh']['a_i'], da_pi)

        # NEW: solving for transfers and effect on distribution
        dWa = ss.internals['hh']['Pi'] @ ((c - ss.internals['hh']['c']) / h * upp)
        curlyWa[s] = dWa
        dT_pe = -dWa / (R*Waa)
        if no_con:
            dT_pe[:, 0] = 0  # if you're at constraint, can't get insurance
        dlambda = np.vdot(Dbeg, dT_pe) / Lambda
        dT = dT_pe + dlambda * sensitivity
        dT = R * dT # scaling by R so that we can transfer to post-return wealth
        curlyD_corr[s] = ss.internals['hh']['Pi'].T @ local_shock(Dbeg, dT / agrid_diff_aug) 
        curlyT[s] = R*dT # NOTE: Scaling by R here so that we have transfers to post-return wealth
        curlylambda[s] = dlambda
        
    return curlyY, curlyD, curlyD_corr, curlyT, curlyWa, curlylambda


def prelim_ss(ss):
    """Preliminary steady state calculations needed for correction in step 1"""
    agrid_diff = np.diff(ss.internals['hh']['a_grid'])
    apol_diff = agrid_diff[ss.internals['hh']['a_i']]
    agrid_diff_aug = np.append(agrid_diff, agrid_diff[-1])

    R = 1 + ss['r']
    up = ss.internals['hh']['c'] ** (-1/ss['eis'])
    upp = -1/ss['eis'] * up / ss.internals['hh']['c']
    mpcs = get_mpcs(ss)
    Wa = ss.internals['hh']['Pi'] @ up
    Waa = ss.internals['hh']['Pi'] @ (mpcs * upp)
    Dbeg = sim.forward_policy(ss.internals['hh']['D'], ss.internals['hh']['a_i'], ss.internals['hh']['a_pi'])
    return R, Waa, Wa, upp, Dbeg, apol_diff, agrid_diff_aug


def J_from_F(F):
    """Recursion J(t,s) = J(t-1,s-1) + F(t,s) to build Jacobian J from fake news F"""
    J = F.copy()
    for t in range(1, F.shape[0]):
        J[1:, t] += J[:-1, t-1]
    return J


"""New helper functions"""

@njit
def forward_policy_shock(Dss, a_i, da_pi):
    dD = np.zeros_like(Dss)
    for e in range(a_i.shape[0]):
        for a in range(a_i.shape[1]):
            chg = da_pi[e,a]*Dss[e,a]
            dD[e, a_i[e,a]] -= chg
            dD[e, a_i[e,a]+1] += chg
    return dD

@njit
def local_shock(Dss, da_pi):
    dD = np.zeros_like(Dss)
    for e in range(Dss.shape[0]):
        for a in range(Dss.shape[1]):
            chg = da_pi[e,a]*Dss[e,a]
            if a < Dss.shape[1]:
                dD[e, a] -= chg
                dD[e, a+1] += chg
            else:
                dD[e, a-1] -= chg
                dD[e, a] += chg
    return dD

def get_mpcs(ss):
    c, a_grid, a, r = ss.internals['hh']['c'], ss.internals['hh']['a_grid'], ss.internals['hh']['a'], ss['r']
    mpcs = np.empty_like(c)
        
    # symmetric differences away from boundaries
    mpcs[:, 1:-1] = (c[:, 2:] - c[:, 0:-2]) / (a_grid[2:] - a_grid[:-2]) / (1+r)

    # asymmetric first differences at boundaries
    mpcs[:, 0]  = (c[:, 1] - c[:, 0]) / (a_grid[1] - a_grid[0]) / (1+r)
    mpcs[:, -1] = (c[:, -1] - c[:, -2]) / (a_grid[-1] - a_grid[-2]) / (1+r)

    # special case of constrained
    mpcs[a == a_grid[0]] = 1

    return mpcs