## SLAP2 DMD volume alignment

This repository supports alignment of structural reference volumes collected from the two SLAP2 DMD imaging paths. The immediate goal is to combine partially overlapping DMD1 and DMD2 z-stacks into a shared anatomical reference volume suitable for dendritic tracing and eventual synapse registration.

Typical input data are motion-corrected structural reference stacks from DMD1 and DMD2. These may contain one or two imaging channels, may differ in z-spacing, and usually contain partial axial overlap. In current VIP Synaptic Dynamics datasets, each stack is often ~150 µm deep with ~1.5 µm z-steps and ~20–50 µm overlap between DMDs.

### Metadata transforms

SLAP2 `.meta` files contain a machine-configuration object graph. For the example structure volumes from `20260227_150709`, the DMD-to-sample transforms were found under:

```text
/#refs#/P1/dmdPixel2SampleTransform   # DMD1
/#refs#/Q1/dmdPixel2SampleTransform   # DMD2