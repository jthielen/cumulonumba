# Cumulonumba: moist thermodynamic and convective parameters implemented with Numba

*This project is in a proof-of-concept/pre-alpha state, based on the immediate research needs of @jthielen for fast CAPE calculations compatible with Awkward Arrays. Caveat emptor!*

More details to come, but suffice to say for now that this is a collection of MetPy calculations ported to Numba JIT routines with all the Pint- and xarray-specific handling stripped out, and top-level interfaces redesigned for "many-profile" style data structures.
 
## Dev Notes

### Overall Plan/Brainstorming

- Goal is to have at least CAPE, CIN, and SRH implemented in numba and runnable on Awkward Array (as I would get out of a parquet store)
- Should be easy though, once having that, to also have a version for grids (as a demo)
- Not sure on API though, given that your "awkward array" version will likely be structured different from a gufunc/grid version?
- there will be lots of copy-from-metpy, remove-unit-stuff (esp. tests!)