The evaluation of diffusion based phyiscs aware generative AI world models (buzzwords holy) is hard. 
We can firstly begin by focusing on a few key pillars that we can later expand upon with additional metrics. 

### Alignment with the original simulation and control inputs

Since the goal of the pipeline is to *transfer* the simulated video into a realistic scenario, we need to test how well the output actually aligns with the original simulation. 

To ensure that this is the case during generation, we apply certain control inputs to modulate and condition the diffusion model to generate certain behaviours. Specifically, for COSMOS we apply:
- Segmentation
- Depth
- Edge

By taking the ground-truth semantic data from CARLA, we can check that key objects (which are scenario-critical) are still present. This can be derived from the cityscapes paper which classes objects as either related to the scenario or purely background. 

So, for each class we:
- generate a ground truth mask from CARLA
- using YOLO for basic bounding box detection + SAM2 or DINOv3 for mask refinement 
- calculate the per class IoU, which tells us how much of the total prediction area (for both masks) overlaps with the actual prediction (Area of Overlap / Area of Union).

Calculate these metrics over the entire video (distribution).
 
### Image quality

As a basic measure of structure preservation, we can apply SSIM (Structural Similarity Image Metric). This calculates whether the image that has been transferred is structurally consistent with the original simulation. This needs to be investigated further as its biased towards appearance, but could give a good indicator of the visual quality and changes applied by the world model. We know a score of 1.0 is impossible, as that would simple be an identical image to the simulator. 

FVD between real video and CARLA RGB? Again, good to compute but need to fully understand what it achieves.
Learned Perceptual Image Patch Similarity - measures how similar two images look to a human. Can be good to compute the metric, but need to understand what it really does. 

### Motion and temporal Consistency (Including Trajectory and Traffic Rules)

Optical‑flow warp error 
TL Transition Consistency: JS divergence between CARLA and Cosmos TL transition matrices.
FTD - Fréchet Trajectory Distance. Important as the trajectory must match due to the transfer.

These would need to be applied no only to the ego, but also for each vehicle present in the scenario. 

### Overall Plan

Overall, set of metrics is rich, but we need to identify the ones that are important specifically for transfer. I think the best bet is to implement a basic pipline where:

- We take a generated dataset and construct a class with frame buffers for each control input + RGB input + final output. 
- Write a abstract class which can be used to implement full evaluations - inputs are frame buffers from the exact controls that are needed. Each script is responsible for each full setup + evaluation. 
- Split evaluations into groups
    - visual
    - motion and trajectories
    - semantic
    - traffic signals

### Focus for Tomorrow

Implement the overall pipeline - no need to worry about it being very flexible atm, just working well enough to implement 2-3 metrics. 

1. SSIM for image quality 
2. FVD for image quality
3. YOLO10 + SAM2 (IoU) for semantics
4. Optical‑flow warp error for motions
5. FTD for single vehicle (EGO)


### Potential blockers for the future:

- Multiple Camera views needed - how do we evaluate across those?
- How do we generate multiple camera views?
    - Using a new pipeline which needs a World Scenario Description (More work)?
    - Multiple Cosmos Runs?
- Suggest potential idea of splitting this into two papers
    - First paper introduces a pipeline for style transfer + offline evaluation in CARLA
    - Second builds on this for fidelity optimisation
- This is a lot of work

For references see:
https://www.perplexity.ai/search/0c8363a1-4450-4c2c-80d4-917f89e6248c
