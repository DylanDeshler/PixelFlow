# JiT-B with 8px patches: 4x the tokens of JiT-B/16 at the same resolution (256 tokens at 128px, a 16x16 grid). The dense
# ceiling of the fixed-budget quadtree study (research_proposal_fixed_budget_quadtree_patch_allocation.md), and the same
# token count JiT-B/16 has at 256px. Patches are 192 values, still compressed by the 128-d bottleneck.
# About 4-5x the compute and activations of JiT-B/16 per image; if batch 1024 doesn't fit, keep the effective batch with
# --batch_size=512 --gradient_accumulation_steps=2 (same lr, same run directory).
# Compose with a dataset tier: python train_jit.py config/jit_in1k_128.py config/model_jit_b8.py ...  (-> runs/in1k_128/jit-b16_p8_icl0/...)
model = 'JiT-B/16'
patch_size = 8
