

import numpy as np

def calc_drag(bead_radius, dist_beadcent_wall, prints=True):
    '''
    bead_radius : [m]
    dist_beadcent_wall : distance center of bead to wall, careful to be >= bead_radius

    return gamma (6 pi eta radius), gamma_parallel, gamma_perp, in [Pa s]
    '''
    #if dist_beadcent_wall < bead_radius:
    #    dist_beadcent_wall = bead_radius
    eta = 0.001 # [Pa*s]=[N*s/m^2]]     water viscosity
    a_s = np.clip(bead_radius/dist_beadcent_wall, 0,0.85)
    gamma = 6*np.pi*eta*bead_radius
    gamma_parallel = gamma/(1 - (9/16)*(a_s) + (1/8)*(a_s)**3)
    gamma_perp     = gamma/(1 - (9/8 )*(a_s) + (1/2)*(a_s)**3)
    if prints:
        print(f'gamma          = {gamma}')
        print(f'gamma_parallel = {gamma_parallel}')
        print(f'gamma_perp     = {gamma_perp}')

    return gamma, gamma_parallel, gamma_perp
