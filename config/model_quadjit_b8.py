# Fixed-budget quadtree tokenization (model_quadtree.py, research_proposal_fixed_budget_quadtree_patch_allocation.md):
# JiT-B's backbone on exactly quad_tokens multi-scale patches per image (32 / 16 / 8 px at 128px), placed by a router.
# Compose with a dataset tier and pick the router / budget on the command line, e.g.
#   python train_jit.py config/jit_in1k_128.py config/model_quadjit_b8.py --optimizer_type=muon --attn_res=full --adaln_single=True \
#       --quad_router=dino --quad_route_baseline=<frozen uniform JiT checkpoint> --quad_tokens=64
# (-> runs/in1k_128/quadjit-b8/muon_attnres-full_adalnsingle_route-dino/...). See README "Fixed-budget quadtree tokenization".
model = 'QuadJiT-B/8'
