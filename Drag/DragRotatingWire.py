def drag_wire(L=1e-6, D=7e-9):
    '''drag of a wire, length L, diam D
    from Dynamics of paramagnetic nanostructured rods under rotating field 2011'''
    import numpy as np
    eta = 0.001 # Pa s, water viscosity
    gamma = np.pi*eta*L**3 / (3*np.log(L/D)-2)
    gamma = gamma *1e21 # pN nm sec
    print('gamma = '+str(gamma)+' pN nm sec')
    
