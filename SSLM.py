import numpy
from sklearn.metrics.pairwise import *
import numpy as np
import numpy.linalg as la
import numpy.random as rn
import cvxopt
from math import *
import scipy.optimize

def gen_diadig(vec):
    n = vec.size
    Y = np.array([vec])
    tmp = np.repeat(Y, n, 0)
    ymat = tmp * tmp.T
    return ymat



class SSLM:

    def __init__(self, X, y, kern, nu = 1.0, nu1 = 0.2, nu2 = 0.2, proba = False,start=None):
        self.X = X
        self.y = y
        self.kern = kern
        self.nu = nu
        self.nu1 = nu1
        self.nu2 = nu2

        self.N = len(y)
        self.m1 = (self.N + sum(y))/2
        self.m2 = self.N - self.m1
        self.proba = proba
        self.start = start


        self.R, self.rho, self.cc, self.a_lst, self.idxes_SVp, self.idxes_SVn\
                = self._compute_important_parameters()

        if proba:
            self.A_sigmoid, self.B_sigmoid = self._fit_sigmoid()
        else:
            self.A_sigmoid, self.B_sigmoid = None, None

    def predict(self, x,index=0):
       # print('this is line {}',x.shape,' index:',index)
        # 1000,2   x_temp (1,2)    self.X.T   (1000,2)
        # 100,96   x_temp (1,96)    self.X.T  (100,96)
        x_temp = x.reshape(1,-1)
        val = (self.R**2 - self.cc - self.kern(x_temp, x_temp) + sum(2 * self.kern(self.X.T, x_temp).flatten() * self.a_lst * self.y)).item()
        return val

    def predict_proba(self, x):
        prob = 1.0/(1 + exp(self.A_sigmoid * self.predict(x) + self.B_sigmoid))
        return prob

    def _compute_important_parameters(self):
        gram, gramymat, a_lst = self._solve_qp()

        eps = 0.00001
        idxes_S1 = []
        for i in range(int(self.m1)):
            a = a_lst[i]
            if eps < a and a < 1.0/(self.nu1 * self.m1) - eps:
                idxes_S1.append(i)

        idxes_S2 = []
        for i in range(int(self.m1), int(self.m1) + int(self.m2)):
            a = a_lst[i]
            if eps < a and a < 1.0/(self.nu2 * int(self.m2)) - eps:
                idxes_S2.append(i)
        print(idxes_S1)
        print(idxes_S2)

        cc = sum(sum(gen_diadig(a_lst) * gramymat))
        f_inner = lambda idx: gram[idx, idx] - sum(2 * gram[idx, :] * a_lst * self.y) + cc

        P1 = sum(map(f_inner, idxes_S1))
        P2 = sum(map(f_inner, idxes_S2))

        n1 = len(idxes_S1)
        n2 = len(idxes_S2)
        print (P1)
        print (P2)
        print(n1)
        print(n2)
        R = sqrt(P1/n1)
        print(P2/n2)
        print(P1/n1)
        rho = sqrt(abs(P2/n2 - P1/n1))
        return R, rho, cc, a_lst, idxes_S1, idxes_S2

    def _solve_qp(self):
        Ymat = gen_diadig(np.array(self.y))

        gram = self.kern(self.X.T)
        gram_diag_matrix = np.diag(gram)

        #  gram =  1000*1000    Ymat  = 1000*1000
        gramymat = gram * Ymat
        gramymat_diag = np.array([-gram_diag_matrix]).T * np.array([self.y]).T

        P = cvxopt.matrix(gramymat)
        q = cvxopt.matrix(gramymat_diag)

        # eq 15
        A_15 = np.array([self.y],dtype = np.float64)
        b_15 = np.eye(1)
        A_16 = np.ones((1, self.N))
        b_16 = np.eye(1)*(2 * self.nu + 1)
        A_ = np.vstack((A_15, A_16))
        B_ = np.vstack((b_15, b_16))
        A = cvxopt.matrix(A_)
        b = cvxopt.matrix(B_)

        G0 = np.eye(self.N)
        G1 = - np.eye(self.N)
        G_ = np.vstack((G0, G1))
        G = cvxopt.matrix(G_)

        h0p = np.ones(int(self.m1))/(self.nu1 * int(self.m1))
        h0m = np.ones(self.N - int(self.m1))/(self.nu2 * (self.N - int(self.m1)))
        h1 = np.zeros(self.N)
        h_ = np.block([h0p, h0m, h1])
        h = cvxopt.matrix(h_)
        sol = cvxopt.solvers.qp(P, q, A=A, b=b, G=G, h=h)
        a_lst = np.array([sol["x"][i] for i in range(self.N)])
        return gram, gramymat, a_lst

    def _fit_sigmoid(self):
        # notation is similar to platt's paper
        t_pos = (int(self.m1) + 1.0)/(int(self.m1) + 2.0)
        t_neg = 1.0/(int(self.m2) + 2.0)
        logical = (self.y > 0)
        t_vec = logical * t_pos + ~logical * t_neg
        f_vec = np.array([self.predict(self.X[:, i],i) for i in range(self.N)])

        def func(params):
            A, B = params
            probs = 1.0/(1.0 + np.exp(A * f_vec + B))
            values = - (t_vec * np.log(probs) + (1 - t_vec) * np.log(1 - probs))
            val = sum(values)
            return val

        sol = scipy.optimize.minimize(func, [0, 0], method = "powell")
        A_sigmoid, B_sigmoid = sol.x
        return A_sigmoid, B_sigmoid

