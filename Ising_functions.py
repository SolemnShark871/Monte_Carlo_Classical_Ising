#Import essential modules
import numpy as np  # package for arrays
import matplotlib.pyplot as plt  # package for plotting
import time  # for timing
import random #generating random numbers
import math
import csv #for storing and retrieveing data
%matplotlib inline
plt.style.use("seaborn-dark")
plt.style.use("default")
#### for nDim dimensions #####

def init(L,nDim):
        N = L**nDim #number of lattice sites for a hypercubic lattice of dimension nDim 
        state = 2 * np.random.randint(2, size=(N,1)) - 1
        return state

def index(L,nDim):
    
        N = L**nDim
        ns = np.transpose(np.linspace(0,N-1,N))
        Indices = np.zeros((N,2*nDim))

        for ndim in range(1,nDim+1):
            rangminus = np.linspace(0,L**(ndim-1)-1,num = L**(ndim-1))
            rangplus = np.linspace(L**ndim - L**(ndim-1) ,L**(ndim)-1, num = L**(ndim-1))

            for n in range(N):
                 if ndim ==1:
                    if n%L == 0: 
                        #print("{} has a boundary in the negative direction".format(i))
                        nminus = n + L -1
                    else: 
                        nminus = n -1
                    if n%L == L-1:
                        #print("{} has a boundary in the negative direction".format(i))
                        nplus = n - L + 1
                    else:
                        nplus = n + 1
                 else: #only works for ndim >1
                        result = (L**ndim- n - 1)%L**(ndim)
                        if result in rangplus:
                            nminus = n + (L-1)*L**(ndim - 1)
                        else: 
                            nminus = n - L**(ndim-1)
                        if result in rangminus:
                            nplus = n - L**(ndim - 1)*(L-1)
                        else: 
                            nplus = n + L**(ndim-1)
                 Indices[n,2*(ndim-1)] = int(nminus)
                 Indices[n,2*ndim-1] = int(nplus)

        return(Indices)

#The Blocking Method Code mine & Erik

def BlockAv(data,Nb):  #Nb is the number of blocks
    
    # Function that calculates the error bar on the Nb blocks of a dataset with Ndata correlated data points
    Ndata = len(data) #Number of data points
    nb = Ndata/Nb     #nb is the number of data points per block
    Blocks = np.array_split(data,Nb) #Changed from nb to Nb
    Averages = [np.mean(x) for x in Blocks]
    error = np.std(Averages) /np.sqrt(Nb) #block average = average of all data points #Group suggestion #1: Evaluate sigma using np.std
    return error            #Group suggestion #2: have this algorithnm only return the error

def dE(config,Indices,nDim,n): #Calculates the dE contribution
    En = 0                     #from all the nearest neighbours of lattice site n 
    spins = 0 
    N = len(config)

    for i in range(2*nDim):
        
        j = int(Indices[n,i]) #Returns the lattice site index j of the ith nearest neighbour to the spin site n
        spins = spins + config[j] #adds the spin of the jth lattice site 
    
    En = 2*J*config[n,0]*spins 
    return En

'Total energy for a configuration' #Version that doesn't use dE
def E_dimensionless(config,Indices,L):
    total_energy = 0
    N = len(config)
    for n in range(N):
        s = config[n]
        nb = 0
        for j in range(2*nDim):
            nb += config[int(Indices[n,j])] #the sum of all other spins 
        total_energy += -J*nb * s
    return (total_energy)/(2) #Division by 2 to avoid overcounting and to calculate the energy per lattice site

'Calculate magnetisaton' #Returns the average magnetisation, which will tend to 0 as the number of spins are balanced. 
def magnetization(config):
    Mag = np.sum(config)
    return Mag

def MC_step(config, beta, Energy):
    '''Monte Carlo move using Metropolis algorithm '''
    #Completes N lattice flips 
    N = len(config)
    for i in range(N-1): #This is computationally expensive for larger N
            n = np.random.randint(0, N-1) 
            sigma = config[n]
            del_E = dE(config,Indices,nDim,n)
           # print(del_E)
            if del_E < 0:
                sigma *= -1
                Energy += del_E
            elif random.uniform(0,1) < np.exp(-del_E*beta): 
                sigma *= -1
                Energy += del_E #Energy is not normalised

    return config,Energy

### Write some code to plot the 2D case

def plot_lattice(config):
    N = len(config)
    L = int(np.sqrt(N))
    lattice_2D = np.zeros((L,L))

    for n in range(N):
        i = int((n - (n%L))/L)
        j = n%L
        lattice_2D[i,j] = config[n]

    plt.figure(dpi = 144)
    imgplot = plt.imshow(lattice_2D)
    imgplot.set_interpolation('none')
    plt.xticks(range(L))
    plt.yticks(range(L))
    plt.show()

def Binder_Ratio(A_squared_mean,A):
    Q = A_squared_mean/(A**2)
    return Q

#Important Code: Returns the Observables, the delObservables (naive errors), and errors for a given range of temperatures 
def simulate(config, L,nDim, mcSweeps, eqSteps, err_runs,T,plot_config):
        
    if nDim != 2: #Only 2 dimensional visualisations can be plotted
        plot_config = False
    Indices = index(L,nDim)
    N = L**nDim
   # mcSteps = 1000        # number of MC steps 
    Nb = 10
    T_c = 2/math.log(1 + math.sqrt(2)) #Theoretical Critical Temperature
 
 # initialise all variables
    nt = len(T)
    E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    C_theoric, M_theoric = np.zeros(nt), np.zeros(nt)
    delta_E,delta_M, delta_C, delta_X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    Energies = np.zeros(nt)
    Magnetizations = np.zeros(nt)
    SpecificHeats = np.zeros(nt)
    Susceptibilities = np.zeros(nt)
    
    #Error bars
    Err_E = np.zeros(nt)
    Err_M = np.zeros(nt)
    Err_Cv = np.zeros(nt)
    Err_X = np.zeros(nt)
    Q = np.zeros(nt)
    deltam1 = np.zeros(nt)
    deltam2 = np.zeros(nt)
    ErrQ = np.zeros(nt)
    
    energy =  E_dimensionless(config,Indices,L) #Energy of the initial configuration
    for t in range(nt):
        
        beta = 1./T[t] #T in units of kB
        all_energies = np.zeros(mcSteps) #Save a list of all energies for this timestamp (Later consider replacing with an array)
        all_energies_squared = np.zeros(mcSteps)
        all_mag = np.zeros(mcSteps) 
        all_mag_squared =  np.zeros(mcSteps)

        # evolve the system to equilibrium
        for i in range(eqSteps):
            new_config,new_energy = MC_step(config, beta, energy)
            config,energy = new_config,new_energy
        # list of macroscopic properties
        Ez = np.zeros(err_runs)
        Cz = np.zeros(err_runs)
        Mz = np.zeros(err_runs)
        Xz = np.zeros(err_runs) 

        for j in range(err_runs):
            E = E_squared = M = M_squared = 0
            for i in range(mcSteps):
                config,energy = MC_step(config, beta, energy)
                
                'ATTEMPT TO FIX '
                #OBSERVABLES ARE NORMALISED
                all_energies[i] = energy/N # Divide by the number of lattice sites
                all_energies_squared[i] = all_energies[i]**2
                mag = abs(magnetization(config)) # calculate the abs total mag. at time stamp
                all_mag[i] = mag/N
                all_mag_squared[i] = all_mag[i]**2
                
            # mean (divide by total time steps)
            E_mean = np.average(all_energies)
            E_squared_mean = np.average(all_energies_squared)
            M_mean = np.average(all_mag)
            M_squared_mean = np.average(all_mag_squared)
            Energy = E_mean
            SpecificHeat = N*beta**2 * (E_squared_mean - E_mean**2)
            Magnetization = M_mean
            Susceptibility = N*beta * (M_squared_mean - M_mean**2)
            #Q[t] = Binder_Ratio(M_squared_mean,(M_mean))
            
            Ez[j] = Energy; Cz[j] = SpecificHeat; Mz[j] = Magnetization; Xz[j] = Susceptibility;
            
        if plot_config == True:
            plot_lattice(config) #remove this later
            print('Temperature = {}'.format(round(T[t],3)))
            print('Magnetisation = {}'.format(round(Magnetization,3)))

        Energies[t] = np.mean(Ez)/2 #Division by 2 for overcounting sites
        Err_E[t] = float(BlockAv(all_energies,Nb))
        Err_M[t] = float(BlockAv(all_mag,Nb))

        #Section dedicated to Cv error
        deltx = BlockAv(all_energies_squared,Nb) #remade using BlockAv
        delty = BlockAv(all_energies,Nb)
        Err_Cv[t] = float((beta**2)*np.sqrt(deltx**2 + delty**2))

        #Section dedicated to X error
        deltam1[t] = BlockAv(all_mag,Nb)
        deltam2[t] = BlockAv(all_mag_squared,Nb)

        Err_X[t] = float((beta)*np.sqrt(deltam2[t]**2+deltam1[t]**2)) #Pietro Method
      
        Magnetizations[t] = np.mean(Mz)
        SpecificHeats[t] = np.mean(Cz)  
        Susceptibilities[t] = np.mean(Xz) 
        Q[t] = Binder_Ratio(M_squared_mean,(M_mean))
       
        #ErrQ[t] = (1/M_squared_mean)*np.sqrt(deltam1[t]**2+(Q[t]*deltam2[t])**2)
        ErrQ[t] = np.sqrt((deltam2[t]/M_squared_mean)**2+(deltam1[t]/M_mean)**2)
    
    return (Energies,Magnetizations,SpecificHeats,Susceptibilities,Err_E,Err_M,Err_Cv,Err_X,Q,ErrQ)