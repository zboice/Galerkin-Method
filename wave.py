import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as plt_ani

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
        return 2*Xs*Galerkin.cheb_T(n-1, Xs) - Galerkin.cheb_T(n-2, Xs)

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

    #outlined on pg 29
    def homogeneous_cheb(n, a, Xs):
        return Galerkin.cheb_T(n, Xs) - Galerkin.cheb_T(n%2, Xs)
        
    def homogeneous_cheb_deriv(n, a, Xs):
        if n%2 == 0:
            return n*Galerkin.cheb_U(n-1, Xs)
        return n*Galerkin.cheb_U(n-1, Xs) - np.ones_like(Xs)

    def basis_init(self, inhomogeneous, off, c0, c1, c2, c_non_lin):
        if self.family == 'sines':
            self.norm = 2/a
            self.nstart = 1
            self.bas = Galerkin.sines
            self.bas_der = Galerkin.sine_deriv
            self.inho = lambda n, t: 1 #np.exp(-10j*t)*inhomogeneous
            self.operator = lambda n, m: (c0*self.bas(n, a, self.coord)*self.bas(m, a, self.coord) 
                                      + c1*self.bas(n, a, self.coord)*self.bas_der(m, a, self.coord) 
                                      + c2*self.bas_der(n, a, self.coord)*self.bas_der(m, a, self.coord)
                                      + c_non_lin*self.bas(n, a, self.coord)**2*self.bas_der(m, a, self.coord))
            
        elif self.family == 'cheb':
            self.norm = 1/np.sqrt(1-self.coord**2)
            self.nstart = 0
            self.bas = Galerkin.chebs
            self.bas_der = Galerkin.cheb_deriv

        elif self.family == 'homog_cheb':
            self.norm = (2/np.pi)*1/np.sqrt(1-self.coord**2)
            self.norm_coord = np.linspace(0, np.pi, len(Xs))
            self.nstart = 2
            self.bas = lambda n, a, Xs: Galerkin.homogeneous_cheb(n, a, Xs)
            self.bas_der = lambda n, a, Xs: Galerkin.homogeneous_cheb_deriv(n, a, Xs)

    def __init__(self, bas_fam, inhomogeneous, coordinate_axis, max_basis, times, c0=0, c1=0, c2=0, c_non_lin=0, off=0):
        self.coord = coordinate_axis
        self.num = max_basis
        self.family = bas_fam
        self.times = times
        self.basis_init(inhomogeneous, off, c0, c1, c2, c_non_lin)

    def derivs(self, t, c_vec, answer, h, a, N):
        Neff = N-self.nstart
        hbar = 1; m=1
        M_mat = np.zeros([N-self.nstart, N-self.nstart])
        L_mat = np.zeros_like(M_mat)

        for n in range(self.nstart, N):
            for m in range(self.nstart, N):
                M_mat[n-self.nstart, m-self.nstart] = simpson_int(self.bas(n, a, Xs)*self.bas(m, a, Xs), h)
                L_mat[n-self.nstart, m-self.nstart] = simpson_int(self.bas_der(n, a, Xs)*self.bas_der(m, a, Xs), h)

        answer[:Neff] = c_vec[Neff:]
        answer[Neff:] = np.matmul(np.linalg.inv(M_mat), np.matmul(-L_mat, c_vec[:Neff]))

        return answer

    def initial_approx(self, h, Xs, initial_func):        
        answer = 0
        for n in range(self.nstart, self.num):
            weight = simpson_int(self.bas(n, a, Xs)*initial_func(Xs) * self.norm , h)
            answer += self.bas(n, a, Xs)*weight
    
        return answer

    def initial_cvec(self, answer, u0): # only works for chebyshev polynomials currently
        thetas = self.norm_coord
        htheta =  thetas[1]-thetas[0]
        for n in range(self.nstart, self.num):
            answer[n-self.nstart] = 2/np.pi * simpson_int(u0(np.cos(thetas))*np.cos(n*thetas), htheta)
        return answer

    def runge_kutta(self, u0):
        int_time = range(len(self.times))

        cvecs = np.zeros([len(int_time), 2*(self.num-self.nstart)], complex)
        cvecs[0] = self.initial_cvec(cvecs[0], u0)
        answer = np.zeros_like(cvecs[0])

        a = Xs[-1]
        h = Xs[1]-Xs[0]
        N = self.num
        ht = self.times[1]-self.times[0]

        for j, t in zip(range(len(self.times[:-1])), self.times[:-1]):
            k1 = ht*self.derivs(t, cvecs[j], answer, h, a, N)
            k2 = ht*self.derivs(t+0.5*ht, cvecs[j]+0.5*k1, answer, h, a, N)
            k3 = ht*self.derivs(t+0.5*ht, cvecs[j]+0.5*k2, answer, h, a, N)
            k4 = ht*self.derivs(t+ht, cvecs[j]+k3, answer, h, a, N)

            cvecs[j+1] = cvecs[j] + (k1+2*k2+2*k3+k4)/6
        
        self.int_times = int_time
        self.weights = cvecs[:, :N]
    
    def calc_answer(self):
        answer = np.zeros([len(self.times), len(self.coord)], complex)
        for time in self.int_times:
            sol_vec = self.weights[time]
            for n in range(self.nstart, self.num):
                answer[time] += self.bas(n, a, self.coord)*sol_vec[n-self.nstart]
            
        return answer, self.int_times

def plotAnimated(Nx, x, t, func, ymin=-1, ymax=1, save=False):
    fig = plt.figure()
    axis = plt.axes(xlim=(x[0], x[-1]), ylim=(ymin, ymax))
    line, = axis.plot([], [], lw = 3)

    ani = plt_ani.FuncAnimation(fig, animate, t, fargs=(line, Nx, x, func), blit=True, interval=20)
    plt.show()

    if save:
        writervideo = plt_ani.FFMpegWriter(fps=60)
        ani.save('waves.mp4', writer=writervideo)
        plt.close()

def init(line):
    line.set_data([], [])
    return line,

def animate(t, line, Nx, x, func):
    line.set_data(x, np.real(func[:][t][:2*Nx]))
    return line,

def initial_func(Xs):
    return np.exp(-30*(Xs+a/2*np.ones_like(Xs))**2) 

a = 1
h = 0.5e-2
ht = 0.01
Xs = np.arange(-a+1e-12, a-1e-12, h)

times = np.arange(0, 4, ht) # need to watch a CFL-like condition
N_MAX = 10

eqn = Galerkin('homog_cheb', 1, Xs, N_MAX, times)
eqn.runge_kutta(initial_func)
func, times = eqn.calc_answer()

plt.plot(Xs, initial_func(Xs), label=r'$u_0(x)$')
plt.plot(Xs, func[0], label='Numerical Approximation')
plt.xlabel('x')
plt.legend()
plt.show()

plt.imshow(np.real(func))
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()
plt.show()

plotAnimated(len(Xs), Xs, times, func, -1, 1)