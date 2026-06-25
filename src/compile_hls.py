import keras
import os
import importlib.metadata
from hgq.config import QuantizerConfigScope, LayerConfigScope
from hgq.regularizers import MonoL1
from alkaid.converter import trace_model
from alkaid.codegen import RTLModel

from src.models.jetformer import build_hgq_jetformer

print("--- 1. Native Architectural Reconstruction ---")
quant_scope = QuantizerConfigScope(
    place="all", default_q_type="kbi", overflow_mode="WRAP", br=MonoL1(1e-7)
)
layer_scope = LayerConfigScope(enable_ebops=True, beta0=5e-8)

with quant_scope, layer_scope:
    # Rebuilds the graph natively, bypassing the CLI's IndexError
    model = build_hgq_jetformer(
        in_dim=3,
        embed_dim=64,
        num_heads=2,
        num_classes=5,
        num_transformers=3,
        dropout=0.0,
        num_particles=8,
        activation="ReLU",
        normalization="Batch",
        quantize=True,
    )
    model.load_weights("repositoryModel/KeyError/8_3f.keras")


# ==========================================
# 2. FORCED PLUGIN INITIALIZATION
# ==========================================
print("--- 2. Triggering Alkaid Plugin Loader ---")
# Manually execute the zero-argument callable specified in the docs
# to force HGQ to populate Alkaid's _registry immediately.
for ep in importlib.metadata.entry_points(group="alkaid_keras"):
    if "hgq" in ep.name.lower() or "hgq" in ep.value.lower():
        print(f"[Alkaid] Forcing load of plugin: {ep.name}")
        plugin_callable = ep.load()
        plugin_callable()

# ==========================================
# 3. SURGICAL REGISTRY PATCH
# ==========================================
print("--- 3. Executing Registry Patch ---")
from alkaid.converter.builtin.keras.main import _registry
from hgq.quantizer.quantizer import Quantizer as HGQCoreOp
from hgq.layers import Quantizer as HGQLayer

# Because we forced the plugin to load, HGQLayer is now guaranteed to be in the registry.
if HGQLayer in _registry:
    _registry[HGQCoreOp] = _registry[HGQLayer]
    print("[Alkaid] Successfully bound core Quantizer operation to Layer handler.")
else:
    print("[FATAL] HGQLayer not found. The forced plugin initialization failed.")

# ==========================================
# 4. STATIC DATAFLOW SYNTHESIS
# ==========================================
print("--- 4. Isolated ALIR Trace ---")
# Executing outside the quant_scope prevents the context manager from intercepting the trace.
inp, out = trace_model(model, framework="keras")

print("--- 5. RTL Generation ---")
rtl = RTLModel(inp, out, latency_cutoff=5)
output_dir = "./gen_verilog"
os.makedirs(output_dir, exist_ok=True)
rtl.write(output_dir)

print(f"Synthesis complete. Hardware configuration locked to: {output_dir}")
