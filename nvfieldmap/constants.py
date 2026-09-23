"""Physical constants and default NV parameters.

GAMMA_NV: NV electron gyromagnetic ratio, g * mu_B / h with g = 2.0028
(Doherty et al., Phys. Rep. 528, 1 (2013)) and mu_B/h = 13.996245 GHz/T (CODATA).
The thesis (Eq. 5.1, printed p. 91) uses the rounded value 28 MHz/mT.
"""
G_NV = 2.0028
MU_B_OVER_H_GHZ_PER_T = 13.996245
GAMMA_NV_MHZ_PER_MT = G_NV * MU_B_OVER_H_GHZ_PER_T   # ~28.03; GHz/T == MHz/mT
D_ZFS_MHZ = 2870.0   # nominal room-temperature zero-field splitting
