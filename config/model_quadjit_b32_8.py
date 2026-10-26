# Fixed-budget quadtree tokenization (model_quadtree.py, research_proposal_fixed_budget_quadtree_patch_allocation.md):
# JiT-B's backbone on exactly quad_tokens multi-scale patches per image, placed by a router. The name gives the patch range,
# 32px coarsest to 8px finest (K=64 with the uniform router is uniform 16px patches, K=256 uniform 8px).
# Compose with a dataset tier and pick the router / budget on the command line, e.g.
#   python train_jit.py config/jit_in1k_128.py config/model_quadjit_b32_8.py --optimizer_type=muon --attn_res=full --adaln_single=True \
#       --quad_router=dino --quad_route_baseline=<frozen uniform JiT checkpoint> --quad_tokens=64
# (-> runs/in1k_128/quadjit-b32-8/muon_attnres-full_adalnsingle_route-dino/...). See README "Fixed-budget quadtree tokenization".
model = 'QuadJiT-B/32-8'
