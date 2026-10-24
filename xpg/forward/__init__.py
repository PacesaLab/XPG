"""Pixel-level forward-model integration.

Expected counts of every pixel near predicted reflections, on every frame, from one model, fitted to the raw counts
by Poisson maximum likelihood: mu_pk = B_pk + sum_h I_h f_h(p, k). Neighbours and overlaps share pixels instead of
being masked; the background is a smooth field instead of a ring around each reflection.

    model.py    profile kernel, windows of frames, likelihood passes (scatter / gather)
    fit.py      joint fit of intensities and background per window
    kernels.py  Triton versions of the passes (CUDA)
    driver.py   windows over the sweep, memory budget, results back to the reflections"""
