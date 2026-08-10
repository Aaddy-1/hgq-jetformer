import keras
import hgq

model = keras.models.load_model(
    "/mnt/ccnas2/bdp/as8625/hgq-jetformer/experiment/EXP-18/models/quantized/128_17f.keras",
    compile=False,
)
total_ebops = sum(float(l.ebops) for l in model.layers if hasattr(l, "ebops"))
print(f"EXP-18 Total EBOPs: {total_ebops:,.2f}")
