# Figure and Table Captions

## fig_accuracy_latency_tradeoff.pdf
Accuracy-latency tradeoff on RepDETR3D EVA-02. Full L24 encoding gives the highest accuracy but lowest throughput, naive reuse is fast but degrades sharply, and early-feature memory updates recover most detection quality at much higher FPS. The deterministic L24-L8-L12-L16 cycle illustrates that even a simple policy improves the efficiency frontier.

## fig_drift_predictability_r2.pdf
Grouped scene-level cross-validation for predicting final-representation drift. Online ego-motion metadata is already predictive; oracle annotation metadata improves the upper bound; adding a cheap L8 similarity probe makes both cosine similarity and patch CKA highly predictable before full encoding.

## fig_drift_target_distributions.pdf
Distributions of adjacent-frame L24 drift targets on validation frames. Global cosine similarity has broad support around the chosen easy/medium/hard thresholds, while patch CKA provides a separate local structure-sensitive target that varies substantially across frames.

## fig_l8_probe_vs_final_drift.pdf
Early L8 feature similarity as a cascade signal. The L8 probe is strongly aligned with final L24 cosine similarity and remains informative for patch CKA, supporting a policy that probes L8 before deciding whether to stop at L8, continue to L12, or refresh with L24.

## fig_top_online_metadata_cues.pdf
Top online-safe metadata cues for predicting L24 cosine drift. The dominant signals are ego translation, speed, and yaw-rate; absolute pose features are selected by permutation importance but are weaker as standalone physical explanations.

## table_policy_mlp_classification.pdf
Scene-grouped cross-validation performance for the exported small MLP policy checkpoints. The policy maps easy, medium, and hard frames to L8, L12, and L24 respectively. Offline metadata is an oracle upper bound; online plus L8 cascade is the strongest deployable policy input.
