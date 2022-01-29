# output

Contains output of G-ADSA analysis performed in the notebooks contained in the
`notebooks` folder. Includes processed data tables and plots (`pdf` files).

## File Guide

- `<n>k<f>_<T>c.csv`: data table containing processed (solubility, diffusivity,
  interfacial tension, and specific volume) and raw G-ADSA data for a polyol of
  `<n>` kg/mol molecular weight, functionality `<f>` hydroxyl groups per
  molecule at a temperature of `<T>` degrees Celsius at a range of pressures.
  Below, the headings of the most important properties estimated are provided:
    - Pressure: `p actual [kPa]` (actual measured pressure in kPa)
    - Solubility: `solubility [w/w]`
    - Specific volume: `specific volume (fit) [mL/g]` (this value is computed by
      fitting a model to the drop volume, which smooths out some of the noise)

## Fitting

Ideally, a good fit of this data will fit solubility vs. pressure and
specific volume vs. pressure at both temperatures (30 C and 60 C) for a given
polyol.
We can currently fit solubility vs. pressure well with PC-SAFT, but the
parameters are underdetermined, *i.e.*, many sets of parameters produce a model
that matches the data.
We cannot fit the specific volume vs. pressure with any of those parameter sets,
however, which we attribute to a poor model of association in the polyol.
