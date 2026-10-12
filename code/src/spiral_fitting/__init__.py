"""Diffeomorphic spiral fitting — a global-model alternative to M2+M3.

Implements a scoped-down version of Henderson, "Virtually Unrolling the
Herculaneum Papyri by Diffeomorphic Spiral Fitting" (arXiv:2512.04927,
Dec 2025), which was found and evaluated during this project's real-scroll
testing (see README's PoC Scope section) precisely because it diagnoses and
fixes the failure mode M2's classical connected-component segmentation hit
on real data: "naive use of 3D connected components yields CCs with many
sheets combined, since their segmentation masks touch."

Core idea, unchanged from the paper: instead of trying to get a clean
per-layer *segmentation* at all (M2's approach) and then independently
*flattening* each layer (M3's approach), fit one global parametric model —
an idealized Archimedean spiral warped by a diffeomorphism (a
topology-preserving smooth deformation) — directly to sparse evidence
extracted from the volume. Because the deformation is diffeomorphic, the
fitted result is *guaranteed* to be a single continuous, non-self-
intersecting 2D sheet, even through gaps or touching windings — solving
the layer-jumping problem structurally rather than through segmentation
quality. Flattening falls out of the same fit for free.

What's scoped down from the full paper (see each submodule's docstring for
specifics): the diffeomorphism uses only the flow-field stage (the paper's
own ablation, Table 2, shows this alone gets close to the full model — in
fact slightly *better* MRWD/ChD, at the cost of worse WJF); feature
extraction uses only generic surface paths (no separate fiber-prediction
model, since this project has no trained fiber-segmentation network), so
the fiber-orientation losses aren't included either.
"""
