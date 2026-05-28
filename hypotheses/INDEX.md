# hypotheses/ — INDEX

> 84 committee-grade hypothesis design documents organized into 8
> thematic groups matching `../IDEA_TABLE.md`. Each file follows
> `_TEMPLATE.md`. Every group has its own subdirectory so the flat
> directory is no longer overwhelming.
>
> **Implementation status (May 2026):** 83 of the 84 hypotheses now have
> a shared src module + passing unit tests under
> `../src/nature_inspired_networks/` and `../tests/`; only **H57**
> (audio-cymatic cross-modal) is deferred with no module. 35 tags
> additionally carry a seed-0 CIFAR-10 smoke result. Per-hypothesis
> status lives in `../IDEA_TABLE.md`.

## File-naming convention

`g<N>_<group_name>/H<NN>_<short>.md` where:
- `g<N>` is the group ID 1-8
- `H<NN>` is the zero-padded hypothesis ID
- `<short>` is a kebab-case short name

The full design space + status per hypothesis is in `../IDEA_TABLE.md`.
The master experiment list is in `../EXPERIMENT_LOG.md`.

## Group sizes at a glance

| Group | Subdir | IDs | # docs |
|---|---|---|---|
| G1 Scaling & Growth | `g1_scaling_growth/` | H01–H10 | 10 |
| G2 Layer/Channel/Neuron | `g2_layer_channel_neuron/` | H11–H20 | 10 |
| G3 Topologies & Graphs | `g3_topologies_graphs/` | H21–H30 | 10 |
| G4 Kernels/Attention/Filters | `g4_kernels_attention_filters/` | H31–H40 | 10 |
| G5 Optimization/Init/Reg/NAS | `g5_optimization_init_reg_nas/` | H41–H50 | 10 |
| G6 Topological & Bridging | `g6_topological_bridging/` | H51–H60 | 10 |
| G7 Cross-Paradigm Hybrids | `g7_cross_paradigm_hybrids/` | H61–H75 | 15 |
| G8 Esoteric Extensions | `g8_esoteric_extensions/` | H76–H84 | 9 |
| **Total** | | **H01–H84** | **84** |

## Group G1 — Scaling & Growth (`g1_scaling_growth/`)

10 hypotheses: φ / Fibonacci growth laws applied to depth, width,
resolution, layer counts, parameter budgets, LR schedulers.

| ID | File |
|---|---|
| H01 | `g1_scaling_growth/H01_phi_compound_scaling.md` |
| H02 | `g1_scaling_growth/H02_fibonacci_depth_progression.md` |
| H03 | `g1_scaling_growth/H03_golden_spiral_resolution_scaling.md` |
| H04 | `g1_scaling_growth/H04_phi_self_similar_width.md` |
| H05 | `g1_scaling_growth/H05_fractal_phi_recursion.md` |
| H06 | `g1_scaling_growth/H06_golden_ratio_bottleneck.md` |
| H07 | `g1_scaling_growth/H07_phi_modulated_multi_scale.md` |
| H08 | `g1_scaling_growth/H08_dynamic_phi_growth.md` |
| H09 | `g1_scaling_growth/H09_golden_proportion_param_budget.md` |
| H10 | `g1_scaling_growth/H10_phi_decay_lr_scheduler.md` |

## Group G2 — Layer / Channel / Neuron Architectures (`g2_layer_channel_neuron/`)

10 hypotheses: Fib-sized MLPs, channel counts, neuron connectivity,
skip scaling, head diversity, activation thresholds, ensembles.

| ID | File |
|---|---|
| H11 | `g2_layer_channel_neuron/H11_pure_fibonacci_mlp.md` |
| H12 | `g2_layer_channel_neuron/H12_fib_channel_cnn.md` |
| H13 | `g2_layer_channel_neuron/H13_golden_neuron_connectivity.md` |
| H14 | `g2_layer_channel_neuron/H14_fibonacci_recurrent.md` |
| H15 | `g2_layer_channel_neuron/H15_phi_initialized_embedding.md` |
| H16 | `g2_layer_channel_neuron/H16_fibonacci_head_diversity.md` |
| H17 | `g2_layer_channel_neuron/H17_golden_ratio_skip_connections.md` |
| H18 | `g2_layer_channel_neuron/H18_fib_stage_transition.md` |
| H19 | `g2_layer_channel_neuron/H19_phi_neuron_activation_threshold.md` |
| H20 | `g2_layer_channel_neuron/H20_fibonacci_ensemble.md` |

## Group G3 — Topologies & Graphs (`g3_topologies_graphs/`)

10 hypotheses: hex lattices, toroidal closures, Platonic graphs,
icosahedral / dodecahedral equivariance, fractal toroidal, cymatic hex
resonance, small-world, Platonic-Fib hybrids.

| ID | File |
|---|---|
| H21 | `g3_topologies_graphs/H21_hexagonal_phi_packing.md` |
| H22 | `g3_topologies_graphs/H22_toroidal_phi_closure.md` |
| H23 | `g3_topologies_graphs/H23_platonic_phi_graph.md` |
| H24 | `g3_topologies_graphs/H24_icosahedral_phi_equivariant.md` |
| H25 | `g3_topologies_graphs/H25_dodecahedral_latent.md` |
| H26 | `g3_topologies_graphs/H26_fractal_toroidal.md` |
| H27 | `g3_topologies_graphs/H27_golden_spiral_graph.md` |
| H28 | `g3_topologies_graphs/H28_cymatic_hex_resonance.md` |
| H29 | `g3_topologies_graphs/H29_phi_small_world.md` |
| H30 | `g3_topologies_graphs/H30_platonic_fib_hybrid.md` |

## Group G4 — Kernels / Attention / Filters (`g4_kernels_attention_filters/`)

10 hypotheses: golden-spiral kernels, Fibottention dilated attention,
Vesica Piscis filters, golden-angle RoPE, cymatic wavelets, φ-spiral PE,
pentagonal-symmetric attention, fractal golden filters, harmonic φ
activations, Metatron kernel overlaps.

| ID | File |
|---|---|
| H31 | `g4_kernels_attention_filters/H31_golden_spiral_kernel.md` |
| H32 | `g4_kernels_attention_filters/H32_fibonacci_dilation_attention.md` |
| H33 | `g4_kernels_attention_filters/H33_vesica_piscis_filter.md` |
| H34 | `g4_kernels_attention_filters/H34_golden_angle_rotary.md` |
| H35 | `g4_kernels_attention_filters/H35_cymatic_wavelet.md` |
| H36 | `g4_kernels_attention_filters/H36_phi_spiral_positional_encoding.md` |
| H37 | `g4_kernels_attention_filters/H37_pentagonal_phi_attention.md` |
| H38 | `g4_kernels_attention_filters/H38_fractal_golden_filter.md` |
| H39 | `g4_kernels_attention_filters/H39_harmonic_phi_activation.md` |
| H40 | `g4_kernels_attention_filters/H40_metatron_kernel_overlap.md` |

## Group G5 — Optimization / Init / Regularization / NAS (`g5_optimization_init_reg_nas/`)

10 hypotheses: golden-ratio optimizer, φ weight init, Fib pruning,
golden regularization, sacred NAS, cymatic loss, φ dropout, golden
momentum scheduler, Platonic alignment loss, full sacred hybrid.

| ID | File |
|---|---|
| H41 | `g5_optimization_init_reg_nas/H41_golden_ratio_optimizer.md` |
| H42 | `g5_optimization_init_reg_nas/H42_phi_weight_initialization.md` |
| H43 | `g5_optimization_init_reg_nas/H43_fibonacci_pruning.md` |
| H44 | `g5_optimization_init_reg_nas/H44_golden_regularization.md` |
| H45 | `g5_optimization_init_reg_nas/H45_sacred_nas.md` |
| H46 | `g5_optimization_init_reg_nas/H46_cymatic_loss.md` |
| H47 | `g5_optimization_init_reg_nas/H47_phi_dropout.md` |
| H48 | `g5_optimization_init_reg_nas/H48_golden_momentum_scheduler.md` |
| H49 | `g5_optimization_init_reg_nas/H49_platonic_representation_alignment_loss.md` |
| H50 | `g5_optimization_init_reg_nas/H50_full_sacred_hybrid.md` |

## Group G6 — Topological + Bridging (`g6_topological_bridging/`)

10 hypotheses: Betti loss, drop-path anytime, 2D-3D icosa unfold,
persistent-homology activation reg, Platonic Transformers (Islam 2025),
cymatic dataset, audio-cymatic cross-modal, C4 max→avg pool fix (H58),
trained-feature Betti (H59), 3-seed uncertainty (H60).

| ID | File |
|---|---|
| H51 | `g6_topological_bridging/H51_topological_betti_loss.md` |
| H52 | `g6_topological_bridging/H52_drop_path_anytime.md` |
| H53 | `g6_topological_bridging/H53_icosa_unfold_bridge.md` |
| H54 | `g6_topological_bridging/H54_persistent_homology_activation_reg.md` |
| H55 | `g6_topological_bridging/H55_platonic_transformers.md` |
| H56 | `g6_topological_bridging/H56_cymatic_pattern_dataset.md` |
| H57 | `g6_topological_bridging/H57_audio_cymatic_cross_modal.md` |
| H58 | `g6_topological_bridging/H58_group_avg_pool.md` |
| H59 | `g6_topological_bridging/H59_trained_feature_betti.md` |
| H60 | `g6_topological_bridging/H60_three_seed_uncertainty.md` |

## Group G7 — Cross-paradigm Hybrids (`g7_cross_paradigm_hybrids/`)

15 hypotheses: cross-paradigm fusions between sacred priors and the
Liquid / JEPA / KAN / Transformer / GNN paradigms. H67 is the flagship
full Sacred-Liquid-JEPA-KAN-GNN-Transformer hybrid.

| ID | File |
|---|---|
| H61 | `g7_cross_paradigm_hybrids/H61_sacred_liquid_jepa.md` |
| H62 | `g7_cross_paradigm_hybrids/H62_toroidal_kv_hex_attention.md` |
| H63 | `g7_cross_paradigm_hybrids/H63_platonic_aux_cymatic_teacher.md` |
| H64 | `g7_cross_paradigm_hybrids/H64_dynamic_growth_pruning.md` |
| H65 | `g7_cross_paradigm_hybrids/H65_ph_betti_collapse_loss.md` |
| H66 | `g7_cross_paradigm_hybrids/H66_cymatic_qkv_kernel.md` |
| H67 | `g7_cross_paradigm_hybrids/H67_full_paradigm_hybrid.md` |
| H68 | `g7_cross_paradigm_hybrids/H68_ondevice_world_model.md` |
| H69 | `g7_cross_paradigm_hybrids/H69_kan_metatron_symbolic_head.md` |
| H70 | `g7_cross_paradigm_hybrids/H70_cymatic_low_data_curriculum.md` |
| H71 | `g7_cross_paradigm_hybrids/H71_icosa_rope_3d.md` |
| H72 | `g7_cross_paradigm_hybrids/H72_fractal_vesica_ffn.md` |
| H73 | `g7_cross_paradigm_hybrids/H73_golden_spiral_metatron_pe.md` |
| H74 | `g7_cross_paradigm_hybrids/H74_metatron_overlap_tying.md` |
| H75 | `g7_cross_paradigm_hybrids/H75_harmonic_cymatic_swiglu.md` |

## Group G8 — Esoteric Extensions (`g8_esoteric_extensions/`)

9 hypotheses: neutral recast of the esoteric bonus list
(`../IDEA_TABLE.md` Group G8). Tetrahedral dual-path, 12-fold radial
attention, toroidal latent, morphing-polytope adjacency, constant-width
(Reuleaux) kernel, sinusoidal (SIREN) activation, Voronoi sparse
attention, collapse-gated attention, spectral Hopfield memory. The
esoteric origin is acknowledged in one prose sentence per doc only
(Rule 16); all artifacts are neutral-named.

| ID | File |
|---|---|
| H76 | `g8_esoteric_extensions/H76_tetrahedral_dualpath.md` |
| H77 | `g8_esoteric_extensions/H77_radial_symmetry_12_attention.md` |
| H78 | `g8_esoteric_extensions/H78_toroidal_latent_embedding.md` |
| H79 | `g8_esoteric_extensions/H79_morphing_polytope_adjacency.md` |
| H80 | `g8_esoteric_extensions/H80_constant_width_kernel.md` |
| H81 | `g8_esoteric_extensions/H81_sinusoidal_activation.md` |
| H82 | `g8_esoteric_extensions/H82_voronoi_sparse_attention.md` |
| H83 | `g8_esoteric_extensions/H83_collapse_attention.md` |
| H84 | `g8_esoteric_extensions/H84_spectral_hopfield_memory.md` |

## How files are written

`_TEMPLATE.md` is the contract. Each `H<NN>_<short>.md` fills every
section above its "Status journal" line. The status journal grows
append-only.

The bulk of these files were generated by 5 parallel doc agents
(Doc-Agent-A through Doc-Agent-E) in May 2026. Each agent owned a
disjoint range of IDs and produced its files independently from
the template + `IDEA_TABLE.md` + `EXPERIMENT_LEDGER.md`.
