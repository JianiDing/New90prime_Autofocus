# Tilt + Focus Solver — How Δ_A, Δ_B, Δ_C Are Computed

Reference: [`solve_tilt_focus`](focus_pipeline.py#L702-L838) and the joint variant [`solve_tilt_focus_global`](focus_pipeline.py).

---

## Step 1 — Fit a defocus plane to the FWHM map

Model:

$$
\text{FWHM}^2(x,y) \;=\; \text{FWHM}_0^2 \;+\; \alpha\,\delta z(x,y)^2,
\qquad
\delta z(x,y) \;=\; z_0 + a\,x + b\,y
$$

| Symbol | Meaning |
|---|---|
| $\text{FWHM}_0$ | seeing floor (atmosphere) |
| $\alpha$ | defocus sensitivity |
| $z_0$ | piston (overall focus offset) |
| $a, b$ | tip / tilt slopes across the focal plane |

Free parameters $(\text{FWHM}_0^2,\; \alpha,\; z_0,\; a,\; b)$ are fit by least-squares to the 8 per-amp median FWHM values.

## Step 2 — Evaluate the plane at each actuator

Each actuator A/B/C sits at a known focal-plane position $(x_k, y_k)$ — by default 120° apart on a circle (`_actuator_xy(...)`). The local defocus there is:

$$\delta z_k \;=\; z_0 + a\,x_k + b\,y_k$$

## Step 3 — Correction cancels the local defocus

$$\boxed{\;\Delta_k \;=\; -\delta z_k \;=\; -(z_0 + a\,x_k + b\,y_k)\;}$$

Optimal LVDT setting:

$$\text{LVDT}_k^{\text{opt}} \;=\; \text{LVDT}_k^{\text{current}} + \Delta_k$$

Code:

```python
for name, (ax, ay) in act_xy.items():
    defocus_at_act = z0 + a * ax + b * ay
    corrections[name]  = -defocus_at_act
    optimal_lvdt[name] = current_lvdt[name] - defocus_at_act
```

---

## Geometry intuition

| Case | Result |
|---|---|
| Pure piston ($a=b=0$, $z_0 \neq 0$) | All three Δ's equal → uniform focus shift |
| Pure tilt along $x$ ($a \neq 0$, $z_0=b=0$) | +x actuator gets negative Δ, −x gets positive → mirror tips to flatten |
| General | Linear combination of the two |

---

## Caveats

### 1. Sign degeneracy
Only $\delta z^2$ enters the model, so $(z_0, a, b)$ and $-(z_0, a, b)$ give identical fits. The solver tries both seeds and keeps the lower-residual one, but per-frame sign flips can still happen.  
**Mitigation:** average corrections across several high-R² frames before commanding a move, or use the global fit (below).

### 2. Seeing ↔ piston degeneracy (single-frame fits)
Within one frame, the atmosphere $\text{FWHM}_0$ is the same on all amps, so a uniform increase in seeing looks identical to a uniform piston defocus. The solver can trade them off freely:

| Quantity | Single-frame reliability |
|---|---|
| Tilt slopes $a, b$ | ✅ Robust (from FWHM *differences* between amps) |
| Tilt-only deltas $\Delta_k - \overline{\Delta}$ | ✅ Robust |
| Piston $z_0$ → mean of $(\Delta_A, \Delta_B, \Delta_C)$ | ⚠️ Degenerate with seeing |
| Best-focus LVDT from a single tilt fit | ❌ Don't trust |

**Mitigation:**  
- For **tilt** (relative actuator deltas) — single-frame fits are fine, just average several.  
- For **piston** (overall focus) — use the LVDT-vs-FWHM parabola fit from a focus scan, **not** a single tilt fit.

### 3. Units
Δ comes out in the same units as $z_0$, an *inferred* defocus from FWHM². The code currently assumes 1 defocus unit = 1 LVDT unit. If your encoder calibration differs, multiply Δ by the appropriate conversion factor.

---

## Breaking the degeneracy: `--global-tilt-fit`

For a focus-scan sequence (e.g., 10 exposures stepped in LVDT), use:

```bash
python focus_pipeline.py [...args...] --solve-tilt --global-tilt-fit
```

This runs [`solve_tilt_focus_global`](focus_pipeline.py), which fits a **single** $(\text{FWHM}_0, \alpha)$ jointly across all frames while letting $(z_0, a, b)$ vary per frame. Because seeing must be consistent across the scan but defocus changes between frames, the seeing/piston degeneracy is broken.

Outputs:
- `tilt_global.json` — global summary (`seeing_floor`, `alpha`, `R²_global`, frame list)
- `tilt_result_<num>.json` — per-frame, with `"fit_mode": "global"` flag and shared `seeing_floor` / `alpha`

Required: ≥ 5 frames at varying LVDT for the joint fit to add information.

---

## Recommended workflow

1. **Tonight's focus scan** → 10 exposures at stepped LVDT, same field, same filter.
2. Run pipeline with `--solve-tilt --global-tilt-fit`.
3. **Piston** (overall focus) → take from the LVDT-vs-FWHM parabola fit (`focus_fit.png`, `h` value).
4. **Tilt** (relative actuator deltas) → from `tilt_result_<num>.json`'s `corrections`. Subtract the mean to remove the piston ambiguity:

   ```python
   d = corrections                # {"A":..., "B":..., "C":...}
   mean = (d["A"] + d["B"] + d["C"]) / 3
   tilt_only = {k: d[k] - mean for k in d}
   ```

5. Final command to actuators: piston (from step 3) added to all three, plus tilt-only deltas (from step 4).
