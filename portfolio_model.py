"""
Canonical HANK model with real bonds and stocks

Output of the file is:
- household_ext: household block under exogenous portfolios
- net_wage: wage net of taxes
- mkt_clearing: market clearing conditions
- dividend: dividends
- model_ss: steady state model
"""

import sequence_jacobian as sj  

household = sj.hetblocks.hh_sim.hh

'''Part 1: Wage and market clearing blocks'''

@sj.simple
def mkt_clearing(A, B, Y, C, G, p):
    asset_mkt = A - B - p
    goods_mkt = Y - C - G 
    return asset_mkt, goods_mkt


''' Part 2: Assets '''

@sj.solved(unknowns={'p': (0.1, 25)}, targets=['equity'], solver="brentq")
def equity_price(Z, p, r_ante, mu):
    equity = ( (1-1/mu) * Z(+1) + p(+1)) - p  * (1 + r_ante) 
    return equity

@sj.simple
def asset_returns(Z, p, mu ):
    r_eq = ((1-1/mu) *Z + p) / p(-1) - 1
    r = r_eq
    return r_eq, r

@sj.simple
def asset_pricing_ss(Z, r_ante, mu):
    p = (1-1/mu) *Z/r_ante
    r = r_ante
    r_eq = r_ante
    return p, r, r_eq

'''Part 3: Fiscal block'''

@sj.simple
def fiscal(B, r_ante, G, Y):
    T = (1 + r_ante(-1)) * B(-1) + G - B # total tax burden
    Z = Y - T # disposable income
    return Z, T

@sj.simple
def fiscal_ss(B, G, r_ante, Y, mu):
    T = r_ante*B + G
    Z = Y - T
    w = 1/mu*Z
    return T, Z, w

'''Part 4: Embed HA block'''

def make_grids(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, pi_e, Pi = sj.grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = sj.grids.asset_grid(amin=amin, amax=amax, n=nA)
    return e_grid, pi_e, Pi, a_grid

def wages(Z, e_grid, mu):
    y = 1/mu * Z * e_grid
    return y


'''Part 5: Create the model'''

household = sj.hetblocks.hh_sim.hh
household_ext = household.add_hetinputs([wages, make_grids])
print(household_ext)


model_ss = sj.create_model([household_ext, 
                            mkt_clearing,
                            fiscal_ss,
                            asset_pricing_ss
                          ], name="SS")
