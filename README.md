# ResponseCreator in Python

A script to create a MEGAlib-like Compton Response in Python with arbitrary binning (decoupled binning of $E_m$ and $E_i$, as well as $\phi$, $\nu\lambda$ and $\psi\chi$).

## Use Example

```python
from compton_response import ComptonResponseCreator

Ei_edges = np.linspace(20, 3000, 101)
Em_edges = np.linspace(20, 3000, 101)

stats = ComptonResponseCreator(
    tra_filename   = '/mnt/cluster/rreinhardt/RSL7_ComptonCamera/Mass_Model_old/RSL7_iso.tra.gz',
    Ei_edges       = Ei_edges,
    Em_edges       = Em_edges,
    nside_nulambda = 4,
    nside_psichi   = 4,
    phi_bins       = 18,
    pol_bins       = None,
    dtype          = np.float32,
    max_sq         = 7,
    n_workers      = 30,
    save_path      = '/mnt/cluster/rreinhardt/RSL7_ComptonCamera/Responses/Test.h5',
    overwrite      = True,
)
```

## Parameters

| Parameter        | Description                                                         |
| ---------------- | ------------------------------------------------------------------- |
| `tra_filename`   | `.tra` / `.tra.gz` or nested concat file                            |
| `Ei_edges`       | Bin edges for initial energy [keV]                                  |
| `Em_edges`       | Bin edges for measured energy [keV]                                 |
| `nside_nulambda` | HEALPix NSIDE for source direction $(\nu\lambda)$                  |
| `nside_psichi`   | HEALPix NSIDE for scatter direction $(\psi\chi)$                   |
| `phi_bins`       | Number of Compton-angle bins (0–180°)                               |
| `save_path`      | Output `.h5` path                                                   |
| `pol_bins`       | Polarisation bins; `None` (default) = omit Pol axis                 |
| `max_sq`         | Maximum Compton Sequence length accepted (default: 7)               |
| `overwrite`      | Overwrite existing output file (default: `False`)                   |
| `compress`       | Bitshuffle compression in HDF5 (default: `True`)                    |
| `n_workers`      | `1` = sequential (safe in Jupyter), `N` = parallel worker processes |
| `dtype`          | NumPy dtype for `EFF_AREA` in HDF5 (default: `float32`)             |

## Returns

`stats` — dictionary with the following keys:

* `n_events_total`
* `n_events_good`
* `n_events_bad_angle`
* `n_events_sq_drop`
* `n_events_out_of_range`
* `sparsity`
* `save_path`

## Improvements

* **Arbitrary binning**

  The response can be generated with arbitrary binning, including independently defined binning for $E_i$, $E_m$, $\phi$, $\nu\lambda$, and $\psi\chi$.

* **Flexible input files**

  Concatenation files can be used as input. Nested concatenation files are also supported, allowing multiple simulations to be combined. The code recursively resolves the complete file structure and ultimately processes all individual `.tra` files.

* **Automated effective area calculation**

  The effective area is calculated automatically from the simulation files. The code processes the relevant header and footer information to determine the total number of simulated photons.

* **Parallel processing**

  Multiple worker processes can be used via the `n_workers` parameter. This allows the individual `.tra` files to be parsed in parallel, significantly reducing the processing time for large simulation sets.

## Known Limitations

* **Polarisation**

  Polarisation binning is supported in principle, but this functionality has not yet been tested and should therefore be considered experimental.

* **Small differences in $\phi$ binning**

  In a few edge cases, the Python implementation assigns photons to $\phi$ bins differently from the original C++ implementation in MEGAlib's `responsecreator`. In one test case, for example, 7 out of 26 million photons were assigned to an adjacent $\phi$ bin.
