import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Variables físicas

N = 10
mu = 1

#Creamos una matriz que contenga la energía de cada partícula

def enerMat(M):
    
    n = len(M[0])
    E = np.zeros((n, n))
    
    for i in range(n):
        
        for j in range(n):
            
            if i == n-1 and j == n-1:
                
                sj = M[i-1, j] + M[0, j] + M[i, j-1] + M[i, 0]
            
            elif i == n-1:
                
                sj = M[i-1, j] + M[0, j] + M[i, j-1] + M[i, j+1]
            
            elif j == n-1:
                
                sj = M[i-1, j] + M[i+1, j] + M[i, j-1] + M[i, 0]
                
            else:
            
                sj = M[i-1, j] + M[i+1, j] + M[i, j-1] + M[i, j+1]
                
            E[i, j] = -0.5*sj*M[i, j]
            
    return E

#Esta función calculará la energía en una red de espines

def energia(M):
    
    n = len(M[0])
    E = 0
    Emat = enerMat(M)
    
    for i in range(n):

        for j in range(n):
            
            E += Emat[i, j]
            
    return E/2

#Esta función aplica el método metropolis de montecarlo UNA VEZ

def montecarlo(M, beta=100):
    
    E0 = energia(M)
    x = np.random.choice(range(len(M[0])))
    y = np.random.choice(range(len(M)))
    
    M2 = np.copy(M)
    M2[x, y] *= -1
    
    E = energia(M2)
    dE = E - E0
    
    if dE < 0 or np.random.choice([True, False], p=[np.exp(-dE*beta), 1-np.exp(-dE*beta)]):
        
        return M2
        
    return M

#Iteramos varias veces el método Montecarlo

def montecarloIter(spi, iters, beta=100):
    
    E, M, std, mat = [energia(spi)], [np.mean(spi)], [np.std(enerMat(spi))], [spi]
    
    for i in range(iters):
        
        spi = montecarlo(spi, beta)
        E.append(energia(spi))
        M.append(np.mean(spi))
        
        Emat = enerMat(spi)
        
        std.append(np.std(Emat))
        mat.append(spi)
        
    return E, M, std, mat

#Vamos a definir una función que determine algunos parámetros finales en función de T

def Ef(spi, iters, beta):
    
    E, M, std, mat = montecarloIter(spi, iters, beta)
    
    return E[-1], M[-1], std[-1], mat[-1]

def main():

    #Construimos la matriz de espines
    
    espin = np.zeros((N, N))
    
    for i in range(N):
        
        for j in range(N):
            
            espin[i, j] = np.random.choice([-1, 1], p=[0.5, 0.5])
            
    iters = 2000
    i = [i+1 for i in range(iters+1)]
    
    E, M, std, spi = montecarloIter(espin, iters, beta=100)
    
    plt.figure(figsize=(9, 6))
    plt.plot(i, E)
    plt.xlabel("Número de iteraciones")
    plt.ylabel("$Energía \\ (J)$")
    #plt.savefig("Energía 1.png", dpi=200)
    
    plt.figure(figsize=(9, 6))
    plt.plot(i, M)
    plt.xlabel("Número de iteraciones")
    plt.ylabel("$Magnetización$")
    plt.savefig("Magnetización 1.png", dpi=200)
    
    plt.figure(figsize=(9, 6))
    plt.plot(i, std)
    plt.xlabel("Número de iteraciones")
    plt.ylabel("$\\Delta  E \\ (J)$")
    #plt.savefig("Dispersión 1.png")
    
    plt.figure(figsize=(9, 6))
    plt.imshow(spi[0])
    #plt.savefig("Inicial.png", dpi=200)
    
    plt.figure(figsize=(9, 6))
    plt.imshow(spi[iters-1])
    #plt.savefig("Final.png", dpi=200)
    
    fig = plt.figure()
    ax = plt.axes()
    ax.imshow(spi[0])
    
    def update(g):
        
        ax.imshow(spi[g])
        
    #ani = FuncAnimation(fig, update, frames=[i for i in range(iters)], interval=0.1)
    #ani.save("Animación.gif")
    
    T = np.linspace(0.0001, 4, 50)
    beta = 1/T
    ET = [Ef(espin, iters, i)[1] for i in beta]
    
    plt.figure(figsize=(9, 6))
    plt.scatter(T, ET)
    plt.xlabel("Temperatura")
    plt.ylabel("Magnetización")
    plt.savefig("Cambio de fase.png", dpi=200)
    
if __name__ == "__main__":
    
    main()