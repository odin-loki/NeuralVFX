# Study R: a validated 3D smoke

Status: **in progress**. Audience: dev, research, the owner. A second 3D effect after study N's fire
([STUDY_3D.md](STUDY_3D.md)), done properly this time: the teacher (our 3D solver's smoke) is checked against
established plume physics first (§1), and the learned smoke is then trained with study N2's recipe and tested once
under a rule committed before validation (§2 onwards).

## 1. The teacher against plume physics

### 1.1 The theory and the measurements it is checked against

**Morton, Taylor and Turner (1956)**, "Turbulent gravitational convection from maintained and instantaneous sources",
*Proc. R. Soc. Lond. A* 234, 1-23 (MTT). A plume from a point source of buoyancy flux F (volume flux times reduced
gravity g' = g dT / T, conserved with height in a uniform, still environment) entrains ambient fluid at a rate
proportional to its own vertical velocity (the entrainment coefficient alpha). Its similarity solution, with z measured
from a virtual origin z0:
- the radius grows linearly, **b = (6/5) alpha (z - z0)**;
- the centreline vertical velocity falls as **w ~ F^(1/3) (z - z0)^(-1/3)**;
- the centreline buoyancy falls as **g' ~ F^(2/3) (z - z0)^(-5/3)**, and so does the concentration of a passive tracer
  released with the buoyancy (our soot);
- the buoyancy flux F itself does not change with height.

With top-hat profiles (uniform across radius b_T) the constants are w_T = (5 / (6 alpha_T)) (9 alpha_T F / (10 pi))^(1/3)
z^(-1/3). With Gaussian profiles of equal width (w = W exp(-r^2 / b^2), the 1/e radius b, as MTT and most measurements
use), b = (6/5) alpha_G z, W = (25 F / (12 pi alpha_G^2))^(1/3) z^(-1/3) and G = 2 F / (pi b^2 W); the two descriptions
are the same plume when b_T = sqrt(2) b_G and alpha_T = sqrt(2) alpha_G.

**The entrainment coefficient, measured.** **List (1982)**, "Turbulent jets and plumes", *Annu. Rev. Fluid Mech.* 14,
189-212 (doi:10.1146/annurev.fl.14.010182.001201), reviews the laboratory plumes on which the constant-alpha model rests;
**Carazzo, Kaminski and Tait (2006)**, "The route to self-similarity in turbulent jets and plumes", *J. Fluid Mech.*
547, 137-148, compiles the published data. In van Reeuwijk and Craske's (2015, *J. Fluid Mech.* 782, 333-355) summary of
that compilation, "typical values for the 'top-hat' entrainment coefficient are in the range 0.065 < alpha < 0.080 in
jets and 0.10 < alpha < 0.16 in pure plumes (Fischer et al. 1979; Carazzo, Kaminski & Tait 2006)". This study's
brief gave alpha of about 0.08 to 0.12 for a top-hat plume; we report alpha in both conventions and compare it with
the compiled band 0.10 to 0.16 (top-hat), 0.071 to 0.113 (Gaussian). Carazzo et al. also show that plumes approach
self-similarity only some distance from the source, which matters here: our box is short (§1.4).

**The wind-bent plume.** Far downwind a buoyant plume in a uniform crossflow U is bent over and rises as a line
thermal: **Briggs's two-thirds law**, z = 1.6 F_B^(1/3) x^(2/3) / U (Briggs 1969, *Plume Rise*, USAEC TID-25075;
Briggs 1975, "Plume rise predictions", in *Lectures on Air Pollution and Environmental Impact Analyses*, AMS, 59-111),
with Briggs's flux F_B = F / pi and x the distance downwind.
