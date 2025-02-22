import numpy as np

def myHoughTransform(Im, rhoRes, thetaRes):
    y_vals, x_vals = np.where(Im > 0)
    
    rho_max = int(np.sqrt(Im.shape[0]**2 + Im.shape[1]**2))
    
    thetaScale = np.arange(0, np.pi, thetaRes)
    rhoScale = np.arange(-rho_max, rho_max + rhoRes, rhoRes)
    
    rhos = np.array([x * np.cos(thetaScale) + y * np.sin(thetaScale) for x, y in zip(x_vals, y_vals)])

    ac = np.zeros((len(rhoScale), len(thetaScale)))
   
    for rhos_row in rhos:
        for theta_idx, rho in enumerate(rhos_row):
            rho_idx = np.round((rho + rho_max) / rhoRes).astype(int)
            ac[rho_idx, theta_idx] += 1

    return [ac, rhoScale, thetaScale]
    
