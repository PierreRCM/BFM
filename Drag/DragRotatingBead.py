# finds drag of a bead rotating on an axis offset from its center, + faxen corrections

import numpy as np

def calc_drag(bead_radius=0.25e-6, axis_offset=0.05e-6, dist_beadsurf_glass=0.02e-6, k_hook=400.,prints=1):
    ''' returns the drag (in pN nm s) on a bead of radius "bead_radius" [m], 
    rotating on a circular trajectory whose center is displaced 
    of "axis_offset" [m] from the bead center,
    at a distance from the glass surface (bead surface - glass surface) of
    "dist_beadsurf_glass" [m]
    block polyhooks: <k_hook>=1.52e-12 dyne cm/rad = 152 pN nm/rad
    see: Comparison of Faxen s correction for a microsphere translating or rotating near a surface PRE 2009 '''
	#drag_mot = 2.0e-4          # [pN nm s] /(rad^2?)   drag Motor, from PNAS paper
	#drag_cyl = 5.              # [pN nm s] (rad^2?)    drag cylinder measured 
	#bead_radius = 0.250e-6     # [m]     bead radius
	#axis_offset = 0.050e-6     # [m]     rot.axis offset from center
    #k_hook = 4.0e2             # [pN nm /rad^2],       hook spring constant 

    eta = 0.001                 # [Pa*s]=[N*s/m^2]]     water viscosity
    dist_beadsurf = bead_radius + dist_beadsurf_glass  # [m] dist(bead center, surface)
    # correction beta_parallel (torque)
    faxen_1 = 1 - (1/8.)*(bead_radius/dist_beadsurf)**3 
    # correction gamma_parallel (force):
    faxen_2 = 1 - (9/16.)*(bead_radius/dist_beadsurf) + (1./8)*(bead_radius/dist_beadsurf)**3
    faxen_2_1 = 1 - (9/16.)*(bead_radius/dist_beadsurf) + (1./8)*(bead_radius/dist_beadsurf)**3 - (45/256.)*(bead_radius/dist_beadsurf)**4 - (1./16)*(bead_radius/dist_beadsurf)**5
    
    #drag_bead = 8*np.pi*eta*bead_radius**3/faxen_1 + 6*np.pi*eta*bead_radius*axis_offset**2/faxen_2
    drag_bead = 8*np.pi*eta*bead_radius**3/faxen_1 + 6*np.pi*eta*bead_radius*axis_offset**2/faxen_2_1
    
    drag_bead_pNnms = drag_bead*1e21
    if prints:	    
        print("bead drag : "+str(drag_bead_pNnms)+" pN nm s")
        print("charact.time on hook: "+str(1000*drag_bead_pNnms/k_hook)+" ms")
        print("charact.freq. on hook: "+str(1./(drag_bead_pNnms/k_hook))+" Hz")
        print('faxen 1 = '+str(faxen_1))
        print('faxen 2_1 = '+str(faxen_2_1))
        print('faxen 2 = '+str(faxen_2))
        print('using faxen_2 correction')

    return drag_bead_pNnms
