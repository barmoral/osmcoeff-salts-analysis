"""
Literature curves for NaCl, original (chi = 0) cross interactions - Hosseini & Ashbaugh, JCTC 2023,
19, 8826, SI Tables S3 and S4. NOT part of the analysis code; pass it in when you want the extra curves:

    from literature_reference_nacl_chi0 import REFERENCE
    run_analysis(..., reference=REFERENCE)        # HP_analysis_replicates_thermo.py

molar : (A, B, alpha1, alpha2) in M^-1/2, M^-1/2, M^-1, M^-2   - constant mu_w, A = 1.7964
molal : (A~, B~, alpha~1, alpha~2) in m^-1/2, m^-1/2, m^-1, m^-2 - constant P,  A~ = 1.7937
rho_* : (rho_w0, theta_1, theta_3/2, theta_2) of eq 22, g/cm3 and molality
"""
REFERENCE = dict(
    label="Hosseini & Ashbaugh 2023, $\\chi$=0 (SI S3/S4)",
    molar=(1.7964, 1.9247, 9.3393e-2, 3.4572e-2),
    molal=(1.7937, 1.9218, 9.0418e-2, 1.9887e-2),
    rho_mu=(0.99654, 5.2650e-2, -6.1221e-3, -3.3110e-5),
    rho_P=(0.99654, 5.0615e-2, -5.3907e-3, -6.3600e-4),
)
