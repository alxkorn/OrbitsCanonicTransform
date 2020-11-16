import numpy as np
import re
import pandas as pd
import orbipy as op
import matplotlib.pyplot as plt
import pickle
import sympy as sp
import scipy as scp
import dill
import matplotlib
from copy import copy
from orbipy import mp
import os
from itertools import product

class CT:
    def __init__(self, model):
        self.c2 = self.c(2)
        self.mu = model.mu
        self.g = model.L2 -1 + model.mu #1.-model.mu - model.L1
        self.w1 = sp.sqrt((self.c2 - 2 - sp.sqrt(9*self.c2**2 - 8*self.c2))/(-2))
        self.w2 = sp.sqrt(self.c2)
        self.l1 = sp.sqrt((self.c2 - 2 + sp.sqrt(9*self.c2**2 - 8*self.c2))/2)
        self.s1 = sp.sqrt(2*self.l1*((4 + 3*self.c2)*self.l1**2 + 4 + 5*self.c2 - 6*self.c2**2))
        self.s2 = sp.sqrt(self.w1*((4 + 3*self.c2)*self.w1**2 - 4 - 5*self.c2 + 6*self.c2**2))
    
    def c(self, n):
        g = sp.Symbol('g')
        mu = sp.Symbol('mu')
        return (((-1)**n)*mu+((-1)**n)*((1 - mu)*g**(n + 1))/((1 + g)**(n + 1)))/g**3
#         return (mu+((-1)**n)*((1 - mu)*g**(n + 1))/((1 - g)**(n + 1)))/g**3


    def h(self, n):
        if n<=2:
            raise RuntimeError('n must be > 2')
        x,y,z = sp.symbols('x y z')
        sq = sp.sqrt(x**2+y**2+z**2)
        return self.c(n)*sq**n*sp.together(sp.legendre(n, x/sq))
    
    def R(self):
        return sp.Matrix([[2*self.l1/self.s1,0,0,-2*self.l1/self.s1, 2*self.w1/self.s2,0],
                      [(self.l1**2-2*self.c2-1)/self.s1,(-self.w1**2-2*self.c2-1)/self.s2,0,(self.l1**2-2*self.c2-1)/self.s1,0,0],
                      [0,0,1/sp.sqrt(self.w2),0,0,0],
                      [(self.l1**2+2*self.c2+1)/self.s1,(-self.w1**2+2*self.c2+1)/self.s2,0,(self.l1**2+2*self.c2+1)/self.s1,0,0],
                      [(self.l1**3+(1-2*self.c2)*self.l1)/self.s1,0,0,(-self.l1**3-(1-2*self.c2)*self.l1)/self.s1,(-self.w1**3+(1-2*self.c2)*self.w1)/self.s2,0],
                      [0,0,0,0,0,sp.sqrt(self.w2)]]).subs({'g': self.g, 'mu': self.mu}).evalf()
    
    def symp_change(self):
        x,y,z,px,py,pz = sp.symbols('x1 y1 z1 px1 py1 pz1')
        mat = sp.Matrix([[x],[y],[z],[px],[py],[pz]])
        return self.R()*mat
    
    def h_symp(self, n):
        x, y,z, px, py, pz = sp.symbols('x y z px py pz')
        change = self.symp_change()
        h = self.h(n)
        h = h.subs({'x': change[0], 'y': change[1], 'z': change[2]})
        h = h.subs({'x1': x, 'y1': y, 'z1': z, 'px1': px, 'py1': py, 'pz1': pz})
        h = h.subs({'g': self.g, 'mu':self.mu}).expand().evalf()
        return h
    
    def h_complex(self, n):
        y1,z1,py1,pz1 = sp.symbols('y1 z1 py1 pz1')
        y,z,py,pz = sp.symbols('y z py pz')
        sq2 = math.sqrt(2)
        y_change = (y1 + sp.I*py1)/sq2
        z_change = (z1 + sp.I*pz1)/sq2
        py_change = (py1 + sp.I*y1)/sq2
        pz_change = (pz1 + sp.I*z1)/sq2
        if n == 2:
            h = self.h2_symp()
        elif n>2:
            h = self.h_symp(n)
        else:
            raise RuntimeError('unsupported n')
        h = h.subs({'y': y_change, 'z': z_change, 'py': py_change, 'pz': pz_change}).expand()
        h = h.subs({'y1': y, 'z1': z, 'py1': py, 'pz1': pz})
        return h #self.chop(h)
    
    def gen_func(self, h_comp):
        x, y,z,px,py,pz = sp.symbols('x y z px py pz')
        n1 = self.l1.subs({'g': self.g, 'mu': self.mu}).evalf()
        n2 = sp.I*self.w1.subs({'g': self.g, 'mu': self.mu}).evalf()
        n3 = sp.I*self.w2.subs({'g': self.g, 'mu': self.mu}).evalf()
        pol = sp.Poly(h_comp, x,y,z,px,py,pz)
        mons = pol.monoms()
        gen = 0
        for mon in mons:
            a1 = (mon[3]-mon[0])
            a2 = (mon[4]-mon[1])
            a3 = (mon[5]-mon[2])
            if not (a1==0 and a2==0 and a3==0):
                denominator = a1*n1 + a2*n2 + a3*n3
                sym_part = x**mon[0]*y**mon[1]*z**mon[2]*px**mon[3]*py**mon[4]*pz**mon[5]
                coef = -1*pol.coeff_monomial(mon)
                gen += coef*sym_part/denominator
        return gen.expand()
    
    def pbracket(self, f, g):
        x, y,z, px, py, pz = sp.symbols('x y z px py pz')
        q = [x ,y ,z]
        p = [px, py, pz]
        res = 0
        for i in range(3):
            res += sp.diff(f, q[i])*sp.diff(g, p[i]) - sp.diff(f, p[i])*sp.diff(g, q[i])
        return res.expand()
    
    def h2(self):
        x, y,z, px, py, pz = sp.symbols('x y z px py pz')
        h = (self.c2*(-2*x**2 + y**2 + z**2) + 2*y*px - 2*x*py + px**2 + py**2 + pz**2)/2
        return h
    
    def h2_symp(self):
        change = self.symp_change()
        h = self.h2().subs({'x': change[0], 'y': change[1], 'z': change[2], 'px': change[3], 'py': change[4], 'pz': change[5]})
        x, y,z, px, py, pz = sp.symbols('x y z px py pz')
        h = h.subs({'x1': x, 'y1': y, 'z1': z, 'px1': px, 'py1': py, 'pz1': pz})
        h = h.subs({'g': self.g, 'mu':self.mu}).expand()
        return h #self.chop(h)
    
    def chop(self, h):
        x, y,z,px,py,pz = sp.symbols('x y z px py pz')
        pol = sp.Poly(h, x,y,z,px,py,pz)
        mons = pol.monoms()
        h_new = 0
#         import pdb; pdb.set_trace()
        for mon in mons:
            coef = pol.coeff_monomial(mon)
            coef_chopped = self.chop_coef(coef)
            a, b = coef_chopped.as_real_imag()
            if abs(a)+abs(b) > 0:
                sym_part = x**mon[0]*y**mon[1]*z**mon[2]*px**mon[3]*py**mon[4]*pz**mon[5]
                h_new += coef_chopped*sym_part
        
        return h_new
    
    def chop_coef(self, coef):
        a, b = coef.as_real_imag()
        new_coef = self.chop_num(a) + self.chop_num(b)*sp.I
#         print('Old coef: {}; New coef: {}'.format(coef, new_coef))
        return new_coef

    def chop_num(self, num, tol=1e-14):
        if abs(num) > tol:
            return num
        else:
            return 0
        
    def new_var(self, var, g, n):
        new_var = 0
        prev = 0
        new_var += var
        prev += new_var
        for i in np.arange(1, n+1):
            cur = self.pbracket(prev, g)
            new_var += cur/math.factorial(i)
            prev = cur.copy()
            
        return new_var.expand()#self.realify(new_var)
    
    
    def realify(self, expr):
        y1,z1,py1,pz1 = sp.symbols('y1 z1 py1 pz1')
        y,z,py,pz = sp.symbols('y z py pz')
        sq2 = math.sqrt(2)
        y_change = (y1 - sp.I*py1)/sq2
        z_change = (z1 - sp.I*pz1)/sq2
        py_change = (py1 - sp.I*y1)/sq2
        pz_change = (pz1 - sp.I*z1)/sq2
        
        real_expr = expr.subs({'y': y_change, 'z': z_change, 'py': py_change, 'pz': pz_change})
        real_expr = real_expr.subs({'y1': y, 'z1': z, 'py1': py, 'pz1': pz})
        
        return real_expr.expand()


class CanonicTransform:
    def __init__(self, data_path, model, symp_mat_inverse):
        self.mu = model.mu
        self.gamma = model.L2 -1 + model.mu #1.-model.mu-model.L1
        self.formula = dill.load(open(data_path, "rb"))
        self.symp_mat_inverse = symp_mat_inverse
        
    def symp_change(self, states):
        return (self.symp_mat_inverse @ states.transpose()).transpose()
        
    def apply_shift_scale(self, states):
        # 0-x 1-y 2-z 3-vx 4-vy 5-vz
        shift = (self.mu-1-self.gamma)/self.gamma
        states_new = states/self.gamma
        states_new[:,3] -= states[:,1]/self.gamma
        states_new[:,4] += states[:,0]/self.gamma
        states_new[:,0] += shift
        states_new[:,4] += shift
        return states_new
    
    def symbolic_shift_scale(self, state):
        shift = (self.mu-1-self.gamma)/self.gamma
        new_state = copy(sate)
        for elem in new_state:
            elem = elem/self.gamma
        new_state[3] -= state[1]/self.gamma
        new_state[4] += state[0]/self.gamma
        new_state[0] += shift
        new_state[4] += shift
        return new_state
    
    def apply(self, states):
        arr = self.apply_shift_scale(states)
        arr = self.symp_change(arr)
        result = self.formula(arr[:,0],arr[:,1],arr[:,2],arr[:,3],arr[:,4],arr[:,5])
        return result

class EventQ1(op.base_event):
    def __init__(self, value, terminal, ct):
        self.ct = ct
        super().__init__(value=value, terminal=terminal)
    
    def __call__(self, t, s):
        q1 = self.ct.apply(np.array([s]))[0]
        return  q1 - self.value

class CTPropPlanes:
    def __init__(self, model, precise_model, formula_path, poly_path=None):
        self.model = model
        self.precise_model = precise_model
        ct = CT(model)
        symp_mat_inverse = np.linalg.inv(np.array(ct.R()).astype(np.float64))
        self.ct_x3 = CanonicTransform(formula_path, model, symp_mat_inverse)
        self.log = []
        self.time_log = []
        self.poly_path = poly_path
        
    def target(self, dvy, state):
        s1 = state.copy()
        s1[4] += dvy
        return self.ct_x3.apply(np.array([s1]))[0]
    
    def semi_analytic_correct(self, state):
        # Highly experimental
        if self.poly_path == None:
            return None
        with open(self.poly_path, 'rb') as poly_pickled:
            poly = pickle.load(poly_pickled)
        new_state = list(state.copy())
        dvv = sp.Symbol('dv')
        new_state[4] = dvv
        state_shifted = self.ct_x3.apply_shift_scale(np.array([new_state]))
        st =  sp.Matrix(state_shifted).transpose()
        mat_inv = sp.Matrix(self.ct_x3.symp_mat_inverse)
        res = (mat_inv*st).transpose()       
        sub_hash = {}
        sub_hash['x'] = res[0]
        sub_hash['y'] = res[1]
        sub_hash['z'] = res[2]
        sub_hash['px'] = res[3]
        sub_hash['py'] = res[4]
        sub_hash['pz'] = res[5]
        coeffs = sp.Poly(poly.subs(sub_hash).simplify(), dvv).all_coeffs()
        rs = np.roots(coeffs)
        rs = rs.real[abs(rs.imag)<1e-5]
        
        return rs[np.abs(rs).argmin()]
        
    def correct(self, state):
        bound = self.bound(state)
        dvy = scp.optimize.bisect(self.target,-1.*bound, bound, xtol=1e-14, args=(state,))
        self.log.append(dvy)
        print(dvy)
        return dvy
    
    def bound(self, state):
        bound = 1e-13
        
        for i in range(13):
            if self.bound_crit(state, bound)<0:
                return bound
            else:
                bound*=10
        raise Exception('no root')
            
    def bound_crit(self, state, dv):
        val1 = self.target(dv, state)
        val2 = self.target(-1.*dv, state)
        return val1*val2
    
    def prop(self, s0, q1_min, q1_max, N=1):
        s_init = s0.copy()
        s_init[4] += self.semi_analytic_correct(s_init)
#         s_init[4] += self.correct(s_init)
        q1_left = EventQ1(q1_min, True, self.ct_x3)
        q1_right = EventQ1(q1_max, True, self.ct_x3)
        corr = op.border_correction(self.model, op.y_direction(), [q1_left], [q1_right])
        sk = op.simple_station_keeping(self.precise_model, corr, corr)
        df = sk.prop(0.0, s_init, N=N)
        return df

# Calculate orbit for 100 revolutions and return spacecraft states in DataFrame
def do_calc(job, folder):
    model = op.crtbp3_model()
    precise_model = op.crtbp3_model()
    precise_model.integrator.set_params(max_step=np.pi/180)
    point = model.L2
    
    ctp = CTPropPlanes(model, precise_model, './x3_l2.bin', './x_new_real.pkl')

    # mp.mprint()
    s0 = model.get_zero_state()
    s0[0] = point + job['x']
    s0[2] = job['z']


    tries = 4
    for i in range(1, tries+1):
        try:
            print("try #{}".format(i))
            params = try_params(i)
            df = ctp.prop(s0, params['q_min'], params['q_max'], N=params['nrot'])
        except Exception as e:
            if i < tries:
                print(e)
                continue
            else:
                print("try #{}".format(i+1))
                df_try = ctp.prop(s0, -1.5, 1.5, N=1)
                qs = ctp.ct_x3.apply(df_try.drop('t',axis=1).to_numpy())
                q_min = qs.min() - 0.1
                q_max = qs.max() + 0.1
                df = ctp.prop(s0, q_min, q_max, N=10)
        break


    
    return df

def try_params(try_num):
    par = {1: {'q_min': -1.5, 'q_max': 1.5, 'nrot': 30},
           2: {'q_min': -1., 'q_max': 1.5, 'nrot': 10},
           3: {'q_min': -1.5, 'q_max': 1., 'nrot': 10},
           4: {'q_min': -1., 'q_max': 1., 'nrot': 10}}
    return par[try_num]

# Save orbit DataFrame to pickle (binary) format;
# filename generated using initial x and z coordinates
def do_save(item, folder):
    job = item['job']
    filename = 'orbit_{:.10f}_{:.10f}'.format(job['x'], job['z'])
    # mp.mprint(filename)
    item['res'].to_pickle(os.path.join(folder, filename+'.pkl'))


if __name__ == '__main__':       
    folder = 'calculated_orbits'
    model = op.crtbp3_model()
    EL2_dist = model.L2-(1.-model.mu)
    earth = 1.-model.mu
    
    jobs_todo = product(np.linspace(-1.*EL2_dist, EL2_dist, 300), np.linspace(0, 2600000/model.R, 260))
    # jobs_todo = product(np.linspace(-1.*EL2_dist, EL2_dist, 20), np.linspace(0, 2600000/model.R, 10))
    # jobs_todo = product(np.linspace(-100000/model.R, 100000/model.R, 20), np.linspace(0, 100000/model.R, 10)) # test
    jobs_todo = pd.DataFrame(data=jobs_todo, columns=["x", "z"])
    
    # m = mp('map_1m', do_calc, do_save, folder).update_todo_jobs(jobs_todo)
    m = mp('map_1m', do_calc, do_save, folder)
    m.reset_failed_jobs()

    # add this to csv file:
    # datetime,hash,status
#   test and debug do_calc and do_save functions
    # m.debug_run()

#   run multiprocessed calculation only if debug_run runs without errors  
    m.run(p=4)
    