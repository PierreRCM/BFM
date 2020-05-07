import numpy as np
import matplotlib.pyplot as plt

def drag_cyl(diameter, length, prints=1):
    '''rotational drag of cylinder rotating around its symmetry axis
    from 'Rotational dynamics of rigid, symmetric top macromolecules. Applications to cylinders'
    Tirado, Torre, J.Chem.Phys 73(4) 1980.
    
    diameter in m
    length in m
    return drag in pN nm s
    '''
    A_0 = 3.841
    # dynamic viscosoty of water [N*s/m**2]:
    eta_0 = 1e-3 *1.1
    R = diameter/2.
    L = float(length)
    # parameter p^-1:
    p1 = 2*R/L 
    # values of p^-1 from table II:
    p1_table = np.arange(0, 0.51, 0.05) # is p^-1 = 2R/L
    # delta correction values from table II:
    delta_table = [0, 0.034, 0.067, 0.099, 0.130, 0.159, 0.189, 0.216, 0.243, 0.269, 0.294] 
    # fit of (p1_table, delta_table):
    p1_fit = np.polyfit(p1_table, delta_table, 4)
    # value of delta for R,L given:
    delta_now = np.polyval(p1_fit, p1)
    # drag: 
    csi = (1+delta_now)*(A_0*np.pi*eta_0*L*R**2)
    if prints:
        print('Theoretical rotational drag (around symmetry axis) = '+str(csi*1e21)+' pN nm s')

        plt.figure(1012)
        plt.clf()
        plt.plot(p1, delta_now, 's', ms=9, lw=2)
        plt.plot(p1_table, delta_table, 'o', ms=8)
        plt.xlabel('diameter/length')
        plt.ylabel('gamma correctioni')
        plt.legend(['your cylinder','tabulated values'],loc='upper left')
        plt.grid(1)

    return csi*1e21
