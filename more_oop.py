import numpy as np
import matplotlib.pyplot as plt

def simpson_int(f, h):
    return h/3*(f[0] + f[-1] + 4*sum(f[1:-1:2]) + 2*sum(f[2:-1:2])) 

class Galerkin:
    #info on bases is on 514 of Boyd pdf

    def sines(n, a, Xs):
        return np.sin(n*np.pi*Xs/a)

    def sine_deriv(n, a, Xs):
        return n*np.pi/a*np.cos(n*np.pi*Xs/a)
    
    def cheb_T(n, Xs):
        if n==0:
            return np.ones_like(Xs)
        elif n==1:
            return Xs
        return 2*Xs*Galerkin.cheb_T(n-1, Xs) - Galerkin.cheb_T(n-1, Xs)

    def cheb_U(n, Xs):
        if n==0:
            return np.ones_like(Xs)
        elif n==1:
            return 2*Xs
        return 2*Xs*Galerkin.cheb_U(n-1, Xs) - Galerkin.cheb_U(n-2, Xs)

    def chebs(n, a, Xs):
        return Galerkin.cheb_T(n, Xs)

    def cheb_deriv(n, a, Xs):
        if n==0:
            return np.zeros_like(Xs)
        return n*Galerkin.cheb_U(n-1, Xs)

    def homogeneous_cheb(n, a, Xs):
        if n%2 == 0:
            return Galerkin.cheb_T(n, Xs) - Galerkin.cheb_T(0, Xs)
        return Galerkin.cheb_T(n, Xs) - Galerkin.cheb_T(1, Xs)
    
    def homogeneous_cheb_deriv(n, a, Xs):
        if n%2 == 0:
            return n*Galerkin.cheb_U(n-1, Xs)
        return n*Galerkin.cheb_U(n-1, Xs) - 1*Galerkin.cheb_U(0, Xs)

    def basis_init(self, inhomogeneous, c0, c1, c2, c_non_lin):
        if self.family == 'sines':
            self.nstart = 1
            self.bas = Galerkin.sines
            self.bas_der = Galerkin.sine_deriv
            self.inho = lambda n: inhomogeneous*self.bas(n, a, self.coord)
            self.operator = lambda n, m: (c0*self.bas(n, a, self.coord)*self.bas(m, a, self.coord) 
                                      + c1*self.bas(n, a, self.coord)*self.bas_der(m, a, self.coord) 
                                      + c2*self.bas_der(n, a, self.coord)*self.bas_der(m, a, self.coord)
                                      + c_non_lin*self.bas(n, a, self.coord)**2*self.bas_der(m, a, self.coord))
            
        elif self.family == 'cheb':
            self.nstart = 0
            self.bas = Galerkin.chebs
            self.bas_der = Galerkin.cheb_deriv

        elif self.family == 'homog_cheb':
            self.nstart = 2
            self.bas = Galerkin.homogeneous_cheb
            self.bas_der = Galerkin.homogeneous_cheb_deriv

    def __init__(self, bas_fam, inhomogeneous, coordinate_axis, max_basis, c0=0, c1=0, c2=0, c_non_lin=0):
        self.coord = coordinate_axis
        self.num = max_basis
        self.family = bas_fam
        self.basis_init(inhomogeneous, c0, c1, c2, c_non_lin) # the cn scheme works well enough for linear equations, but sort of fails for non-linear

    def galerkin(self):
        a = Xs[-1]
        h = Xs[1]-Xs[0]
        N = self.num

        mat = np.zeros([N-self.nstart, N-self.nstart])
        f_vec = np.zeros(N-self.nstart)

        for n in range(self.nstart, N):
            for m in range(self.nstart, N):
                mat[n-self.nstart, m-self.nstart] = simpson_int(self.operator(n, m), h)

            f_vec[n-self.nstart] = simpson_int( self.inho(n), h)

        sol_vec = np.matmul(np.linalg.inv(mat), f_vec)

        answer = 0
        for n in range(self.nstart, N):
            answer += self.bas(n, a, self.coord)*sol_vec[n-self.nstart]

        return answer

def inhomogeneous(Xs):
    return -Xs #-(Xs-a/2*np.ones_like(Xs))**2 + (a/2)**2*np.ones_like(Xs)

# trying to find good problem for Chebys
a = 1
k = 3
h = 0.1e-3
Xs = np.arange(0, a, h)

N_MAX = 6
eqn = Galerkin('sines', inhomogeneous(Xs), Xs, N_MAX, c0=-k**2, c1=0, c2=-1, c_non_lin=1)
func = eqn.galerkin()

plt.plot(Xs, func)
#plt.plot(Xs, inhomogeneous(Xs))
#plt.plot(Xs, 1/k**2 * (Xs - 8*np.sin(3*Xs)/np.sin(3)))
#plt.plot(Xs, 1/k**4*(-2+k**2*Xs**2+2*np.cos(k*Xs)-2*np.sin(k*Xs)/np.tan(k)+2*np.sin(k*Xs)/np.sin(k)-k**2*np.sin(k*Xs)/np.sin(k))) # for f(x) = x^2
#plt.plot(Xs, basis(1, a, Xs) + basis(2,a,Xs) + basis(3, a, Xs) + basis(4,a,Xs))
plt.show()