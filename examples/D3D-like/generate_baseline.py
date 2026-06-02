"""Generate the shareable D3D-like baseline equilibrium (D3Dlike_Hmode_baseline.geqdsk).

Self-consistent synthetic DIII-D-like H-mode, built from shot 147131 @ 2300 ms
profiles + boundary at generic round-number targets.  See
D3Dlike_Hmode_baseline_RECIPE.md for the full rationale.

Run:
  N_ISO=16 ISO_W=200 USE_SADDLE=1 SAD_W=200 IP_TARGET=1200000 BT_ABS=2.0 \
  FLIP_BT=1 SR=0.99 SZ=0.97 DZ=0.02 NLEVELS=257 \
  LABEL=D3Dlike_Hmode_baseline python generate_baseline.py
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
sys.path[:] = [p for p in sys.path if not p.rstrip('/').endswith('/plasma/bouquet')
               and not p.rstrip('/').endswith('/plasma/bouquet_phantom_vsc')]
sys.path.insert(0, '/Users/danielburgess/Desktop/plasma/bouquet_coil_bounds')
sys.path.append('/Users/danielburgess/Desktop/plasma/OpenFUSIONToolkit/build_release/python')
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
from bouquet import read_geqdsk, reconstruct_equilibrium
from bouquet.io.pfile import read_pfile

D3DDIR = '/Users/danielburgess/Desktop/plasma/bouquet/examples/D3D-like'
GFILE  = '/Users/danielburgess/Desktop/plasma/CTM-processing/Oak_iterative_beta_scan/g147131.02300_DIIID_KEFIT'
PFILE  = os.path.join(D3DDIR, 'D3Dlike_Hmode_baseline.peqdsk')
MESH   = os.path.join(D3DDIR, 'DIIID_mesh.h5')
NISO   = int(os.environ.get('N_ISO', '16'))
ISOW   = float(os.environ.get('ISO_W', '200'))
LABEL  = os.environ.get('LABEL', 'D3Dlike_Hmode_baseline')
OUT    = os.path.join(D3DDIR, LABEL)
# psi grids
NLEVELS   = int(os.environ.get('NLEVELS', '257'))          # profile-fit grid
# ONE psi padding, used for BOTH the recon solve and save_eqdsk, so the
# equilibrium and its saved g-file stay self-consistent at the LCFS.
PSI_PAD   = float(os.environ.get('PSI_PAD', '1e-4'))
TRUNCATE  = os.environ.get('TRUNCATE', '1') == '1'         # save_eqdsk truncate_eq
ion_N=[6,1,1]; ion_Z=[6,1,1]; ion_A=[12,2,2]

print("[gen147] reading 147131 g-file + D3D-like p-file ...", flush=True)
eqdsk = read_geqdsk(GFILE, cocos=1, nlevels=NLEVELS)
psi_N = eqdsk.psi_N
print(f"[gen147] profile-fit grid: {len(psi_N)} psi_N pts "
      f"({((psi_N>=0.93)&(psi_N<=0.99)).sum()} across pedestal)", flush=True)

# Generic round-number targets: Ip=1.2 MA, |Bt|=2.0 T.  F0>0 for the solve; the
# Bt sign is set to -2.0 AFTER the solve via a Bt-only flip on the saved file.
_ip_t = os.environ.get('IP_TARGET')
if _ip_t is not None:
    _ip_t = float(_ip_t)
    print(f"[gen147] Ip RETARGET: {eqdsk.Ip:.4e} -> {_ip_t:+.4e} A", flush=True)
    eqdsk._raw['CURRENT'] = _ip_t
    if hasattr(eqdsk, '_cache'): eqdsk._cache.clear()
_bt = os.environ.get('BT_ABS')
if _bt is not None:
    F0 = abs(float(_bt)) * eqdsk.R_center
    print(f"[gen147] |Bt| override -> F0={F0:.4f} (|Bt|={_bt} @ R0={eqdsk.R_center:.3f})", flush=True)
else:
    F0 = abs(eqdsk.R_center * eqdsk.B_center)
print(f"[gen147] Ip={eqdsk.Ip:+.4e}  F0={F0:.4f}  li(1)_file={eqdsk.li['li(1)']:.4f}", flush=True)

pf = read_pfile(PFILE)
if pf.ion_species is None: pf.set_ion_species(N=ion_N, Z=ion_Z, A=ion_A)
pf.compute_quasineutrality(); psi_pf, Zeff = pf.compute_zeff()
ne_SI = interp1d(psi_pf, pf.ne*1e20, fill_value='extrapolate')(psi_N)
te_SI = interp1d(psi_pf, pf.te*1e3,  fill_value='extrapolate')(psi_N)
ni_SI = interp1d(psi_pf, pf.ni*1e20, fill_value='extrapolate')(psi_N)
ti_SI = interp1d(psi_pf, pf.ti*1e3,  fill_value='extrapolate')(psi_N)
Zeff_eq = np.clip(interp1d(psi_pf, Zeff, fill_value='extrapolate')(psi_N), 1.0, None)
print(f"[gen147] ne0={ne_SI[0]:.3e}  te0={te_SI[0]:.0f}eV  ti0={ti_SI[0]:.0f}eV", flush=True)

print("[gen147] TokaMaker setup ...", flush=True)
myOFT = OFT_env(nthreads=int(os.environ.get('NTHREADS', '4')))
mygs = TokaMaker(myOFT)
mp, ml, mr, cd, cnd = load_gs_mesh(MESH)
mygs.setup_mesh(mp, ml, mr); mygs.setup_regions(cond_dict=cnd, coil_dict=cd)
mygs.setup(order=3, F0=F0); mygs.settings.maxits = 800; mygs.settings.pm = False; mygs.update_settings()
mygs.set_coil_vsc({'F9A': 1.0, 'F9B': -1.0})
# weak reg toward zero -> free inverse solve
reg = [mygs.coil_reg_term({n: 1.0}, target=0.0, weight=1.0) for n in mygs.coil_sets]
reg.append(mygs.coil_reg_term({'#VSC': 1.0}, target=0.0, weight=1e-2))
mygs.set_coil_reg(reg_terms=reg)

# reshaped boundary (generic look) + 16 even-angle isoflux + saddle X-point
bR0, bZ0 = eqdsk.boundary_R, eqdsk.boundary_Z
cR = 0.5*(bR0.max()+bR0.min()); cZ = float(np.mean(bZ0))
SR=float(os.environ.get('SR','1.0')); SZ=float(os.environ.get('SZ','1.0'))
DR=float(os.environ.get('DR','0.0')); DZ=float(os.environ.get('DZ','0.0'))
bR = cR + SR*(bR0-cR) + DR; bZ = cZ + SZ*(bZ0-cZ) + DZ
R0=0.5*(bR.max()+bR.min()); Z0=float(np.mean(bZ))
ang = np.arctan2(bZ-Z0, bR-R0)
sel = [int(np.argmin(np.abs(np.angle(np.exp(1j*(ang-t)))))) for t in np.linspace(-np.pi,np.pi,NISO,endpoint=False)]
iso = np.column_stack([bR[sel], bZ[sel]]); isow = np.ones(len(iso))*ISOW
mygs.set_isoflux(iso, weights=isow)
print(f"[gen147] isoflux: {len(iso)} pts weight {ISOW}", flush=True)
if os.environ.get('USE_SADDLE','0')=='1':
    SAD_W=float(os.environ.get('SAD_W','200')); ix=int(np.argmin(bZ))
    mygs.set_saddle_constraints(np.array([[float(bR[ix]),float(bZ[ix])]]), np.array([SAD_W]))
    print(f"[gen147] saddle X-point @ R={bR[ix]:.4f} Z={bZ[ix]:.4f} w={SAD_W}", flush=True)

guess = create_power_flux_fun(len(psi_N), 1.5, 1.5)['y']
print("[gen147] reconstruct_equilibrium ...", flush=True)
result = reconstruct_equilibrium(
    mygs, eqdsk, ne_SI, te_SI, ni_SI, ti_SI, Zeff_eq, iso, isow, PSI_PAD,
    guess_jinductive=guess, n_k=5, psi_bridge=0.99, rescale_j_BS=False,
    shelf_psi_N=0.0, initialize_psi=True)
st = mygs.get_stats(lcfs_pad=PSI_PAD, li_normalization='std')
jbs=np.abs(result['j_BS_used']); jphi=np.abs(result['j_phi_fit'])
print(f"[gen147] solved: li(1)={st['l_i']:.4f} q95={st['q_95']:.3f} q0={st['q_0']:.3f} Ip={st['Ip']:.4e}", flush=True)
print(f"[gen147] BOOTSTRAP: j_BS peak={jbs.max()/1e6:.4f}  j_phi peak={jphi.max()/1e6:.4f}  "
      f"ratio={jbs.max()/jphi.max():.3f}", flush=True)

# save: truncate_eq removes the held-then-snapped last separatrix node
mygs.save_eqdsk(OUT+'.geqdsk', nr=257, nz=257, truncate_eq=TRUNCATE, lcfs_pad=PSI_PAD)
print(f"[gen147] save_eqdsk(truncate_eq={TRUNCATE}, lcfs_pad={PSI_PAD})", flush=True)
# COCOS 7->1, then Bt-only flip to land at Bt=-|BT_ABS| (DIII-D convention)
_e1 = read_geqdsk(OUT+'.geqdsk', cocos=7).cocosify(1, copy=True)
if os.environ.get('FLIP_BT','1')=='1':
    _e1._raw['BCENTR']*=-1; _e1._raw['FPOL']=_e1._raw['FPOL']*-1
    if hasattr(_e1,'_cache'): _e1._cache.clear()
_e1.save(OUT+'.geqdsk')
_chk = read_geqdsk(OUT+'.geqdsk', cocos=1)
print(f"[gen147] wrote {OUT}.geqdsk (COCOS 1, Bt={_chk.B_center:+.3f}, Ip={_chk.Ip:+.3e})", flush=True)
# edge derivative report
jt=_chk.j_tor_averaged_direct/1e6; ps=_chk.psi_N; d=np.gradient(jt,ps)
print("[gen147] edge <Jt> derivative (last 4 nodes):", flush=True)
for i in range(len(ps)-4,len(ps)):
    print(f"           psi_N={ps[i]:.4f}  <Jt>={jt[i]:+.4f}  dJt/dpsi={d[i]:+.2f}  FFPRIM={_chk.ffprim[i]:+.4f}", flush=True)
print("[gen147] done.", flush=True)
