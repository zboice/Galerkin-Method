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
            self.inho = lambda n, t: np.exp(-10j*t)*inhomogeneous
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

    def __init__(self, bas_fam, inhomogeneous, coordinate_axis, max_basis, times, c0=0, c1=0, c2=0, c_non_lin=0):
        self.coord = coordinate_axis
        self.num = max_basis
        self.family = bas_fam
        self.times = times
        self.basis_init(inhomogeneous, c0, c1, c2, c_non_lin)

    def derivs(self, t, c_vec, h, a, N):
        M_mat = np.zeros([N-self.nstart, N-self.nstart])
        L_mat = np.zeros_like(M_mat)
        F_vec = np.zeros(N-self.nstart)

        for n in range(self.nstart, N):
            for m in range(self.nstart, N):
                M_mat[n-self.nstart, m-self.nstart] = simpson_int(self.bas(n, a, Xs)*self.bas(m, a, Xs), h)
                L_mat[n-self.nstart, m-self.nstart] = simpson_int(self.bas_der(n, a, Xs)*self.bas_der(m, a, Xs), h)

            F_vec[n-self.nstart] = simpson_int( self.bas(n, a, self.coord)*self.inho(n, t) , h)

        return np.matmul(np.linalg.inv(M_mat), (F_vec - np.matmul(L_mat, c_vec)))

    def initial_cvec(self, answer, c0):
        for n in range(self.nstart, self.num):
            answer[n-self.nstart] = 2/a * simpson_int(self.bas(n, a, self.coord)*c0, h)
        return answer

    def runge_kutta(self, u0):
        int_time = range(len(self.times))
        w = 1

        cvecs = np.zeros([len(int_time), self.num-self.nstart])
        cvecs[0] = self.initial_cvec(cvecs[0], u0)

        a = Xs[-1]
        h = Xs[1]-Xs[0]
        N = self.num
        ht = self.times[1]-self.times[0]

        for j, t in zip(range(len(self.times[:-1])), self.times[:-1]):
            k1 = ht*self.derivs(t, cvecs[j], h, a, N)
            k2 = ht*self.derivs(t+0.5*ht, cvecs[j]+0.5*k1, h, a, N)
            k3 = ht*self.derivs(t+0.5*ht, cvecs[j]+0.5*k2, h, a, N)
            k4 = ht*self.derivs(t+ht, cvecs[j]+k3, h, a, N)

            cvecs[j+1] = cvecs[j] + (k1+2*k2+2*k3+k4)/6
        
        self.int_times = int_time
        self.weights = cvecs
    
    def calc_answer(self):
        answer = np.zeros([len(self.times), len(self.coord)])
        for time in self.int_times:
            sol_vec = self.weights[time]
            for n in range(self.nstart, self.num):
                answer[time] += self.bas(n, a, self.coord)*sol_vec[n-self.nstart]
            
        return answer, self.int_times

def plotAnimated(Nx, x, t, func, ymin=-1, ymax=1):
    #plt.style.use('dark_background')
    fig = plt.figure()
    axis = plt.axes(xlim=(x[0], x[-1]), ylim=(ymin, ymax)) #ylim=(1.1*np.min(func), 1.1*np.max(func)))
    line1, = axis.plot([], [], lw = 3)

    ani = plt_ani.FuncAnimation(fig, animate, t, fargs=(line1, Nx, x, func), blit=True, interval=20)
    plt.show()

def init(line):
    line.set_data([], [])
    return line,

def animate(t, line1, Nx, x, func):
    line1.set_data(x, np.real(func[t][:2*Nx]))
    return line1,

def inhomogeneous(Xs):
    return delta #np.cos(Xs) 

def initial_func(Xs):
    return Xs

a = 1
k = 3
h = 0.1e-2
Xs = np.arange(0, a, h)
times = np.arange(0, 2, 0.01) # need to watch a CFL-like condition

delta = np.zeros_like(Xs)
delta[500:-500] = 3

N_MAX = 6
eqn = Galerkin('sines', inhomogeneous(Xs), Xs, N_MAX, times)
eqn.runge_kutta(Xs)
func, times = eqn.calc_answer()

'''plt.plot(Xs, func[0])
plt.show()'''

plotAnimated(len(Xs), Xs, times, func, -1.25, 1.25)

# 11/19: initial conditions are working now, still goofy though
# Maybe look at some spectre stuff?