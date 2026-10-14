load("@rules_cc//cc:cc_test.bzl", "cc_test")

#ADDED BY JY
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

# TODO: Enbable CUDA build in OSS
# load("@rules_cuda//cuda:defs.bzl", "cuda_library", "requires_cuda")

load("//yggdrasil_decision_forests/utils:compile.bzl", "all_proto_library", "cc_library_ydf")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

config_setting(name = "d2d_direct_on", define_values = {"d2d": "direct"})
config_setting(name = "d2d_kernel_on", define_values = {"d2d": "kernel"})
config_setting(name = "d2d_async_on",  define_values = {"d2d": "async"})
config_setting(name = "d2d_block_stable_on", define_values = {"d2d": "block_stable"})
config_setting(name = "d2d_stable_two_pass_on", define_values = {"d2d": "stable_two_pass"})
config_setting(name = "dedup_bootstrap_on", define_values = {"dedup_bootstrap": "true"})
config_setting(name = "gpu_sample_proj_on", define_values = {"gpu_sample_proj": "true"})
config_setting(name = "bfs_only_on", define_values = {"bfs_only": "true"})
config_setting(name = "use_gpu_on", define_values = {"use_gpu": "true"})
config_setting(name = "no_d2h_on", define_values = {"no_d2h": "true"})
config_setting(name = "use_drsw_on", define_values = {"use_drsw": "true"})
config_setting(name = "node_id_drsw_on", define_values = {"node_id_drsw": "true"})
config_setting(name = "nid_srdw_hist_on", define_values = {"nid_srdw_hist": "true"})
config_setting(name = "srdw_label_hist_on", define_values = {"srdw_label_hist": "true"})
config_setting(name = "symmetric_proj_on", define_values = {"symmetric_proj": "true"})
config_setting(name = "offsets_only_on", define_values = {"offsets_only": "true"})
config_setting(name = "two_pass_on", define_values = {"two_pass": "true"})
config_setting(name = "split_smem_on", define_values = {"split_smem": "true"})
config_setting(name = "verbose_depth_on", define_values = {"verbose_depth": "true"})

# Libraries
# =========================================================

cuda_library(
    name = "oblique_apply_projection",
    srcs = ["oblique_apply_projection.cu"],
    hdrs = ["oblique_apply_projection.cuh", "gpu_timer.h", "node_id_drsw.cuh"],
    includes = ["."],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/status:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/learner/random_forest:random_forest_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
    ],
   copts = [
        "-std=c++17",
        "-Xcompiler",
        "-DGOOGLE_PROTOBUF_NO_ARENA=1",
        "-Wno-deprecated-gpu-targets",
    ] + select({
        ":d2d_direct_on": ["-DD2D_DIRECT=1"],
        ":d2d_kernel_on": ["-DD2D_KERNEL=1"],
        ":d2d_async_on":  ["-DD2D_ASYNC=1"],
        ":d2d_block_stable_on": ["-DD2D_BLOCK_STABLE=1"],
        ":d2d_stable_two_pass_on": ["-DD2D_STABLE_TWO_PASS=1"],
        "//conditions:default": [],
    }) + select({
        ":dedup_bootstrap_on": ["-DDEDUP_BOOTSTRAP=1"],
        "//conditions:default": [],
    }) + select({
        ":gpu_sample_proj_on": ["-DGPU_SAMPLE_PROJ=1"],
        "//conditions:default": [],
    }) + select({
        ":verbose_depth_on": ["-DVERBOSE_DEPTH=1"],
        "//conditions:default": [],
    }) + select({
        ":node_id_drsw_on": ["-DNODE_ID_DRSW=1"],
        "//conditions:default": [],
    }) + select({
        ":nid_srdw_hist_on": ["-DNODE_ID_DRSW_SRDW_HIST=1"],
        "//conditions:default": [],
    }) + select({
        ":srdw_label_hist_on": ["-DNODE_ID_DRSW_SRDW_LABEL_HIST=1"],
        "//conditions:default": [],
    }) + select({
        ":symmetric_proj_on": ["-DSYMMETRIC_PROJ=1"],
        "//conditions:default": [],
    }) + select({
        ":two_pass_on": ["-DTWO_PASS=1"],
        "//conditions:default": [],
    }) + select({
        ":split_smem_on": ["-DSPLIT_SMEM=1"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

cc_library_ydf(
    name = "maximize_groupings",
    hdrs = ["maximize_groupings.h"],
)

cc_library_ydf(
    name = "training",
    srcs = [
        "oblique.cc",
        "oblique_cpu_depthwise_1pass.cc",
        "training.cc",
        "vector_sequence.cc",
    ],
    hdrs = [
        "oblique.h",
        "oblique_cpu_depthwise_1pass.h",
        "oblique_types.h",
        "training.h",
        "vector_sequence.h",
    ],
    deps = [
        ":maximize_groupings",
        ":oblique_apply_projection", #ADDED BY JY
        ":decision_tree_cc_proto",
        ":gpu",
        ":label",
        ":preprocessing",
        ":splitter",
        ":uplift",
        ":utils",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree:training",
        "//yggdrasil_decision_forests/learner/distributed_decision_tree/dataset_cache:dataset_cache_reader",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:bitmap",
        "//yggdrasil_decision_forests/utils:cast",
        "//yggdrasil_decision_forests/utils:circular_buffer",
        "//yggdrasil_decision_forests/utils:compatibility",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:distribution",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:parallel_chrono",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@eigen",
    ] + select({
        "//conditions:default": [
        ],
    }),
    copts = select({
        ":dedup_bootstrap_on": ["-DDEDUP_BOOTSTRAP=1"],
        "//conditions:default": [],
    }) + select({
        ":gpu_sample_proj_on": ["-DGPU_SAMPLE_PROJ=1"],
        "//conditions:default": [],
    }) + select({
        ":bfs_only_on": ["-DBFS_ONLY=1"],
        "//conditions:default": [],
    }) + select({
        ":use_gpu_on": ["-DUSE_GPU=1"],
        "//conditions:default": [],
    }) + select({
        ":no_d2h_on": ["-DNO_D2H=1"],
        "//conditions:default": [],
    }) + select({
        ":use_drsw_on": ["-DUSE_DRSW=1"],
        "//conditions:default": [],
    }) + select({
        ":verbose_depth_on": ["-DVERBOSE_DEPTH=1"],
        "//conditions:default": [],
    }) + select({
        ":d2d_direct_on": ["-DD2D_DIRECT=1"],
        "//conditions:default": [],
    }) + select({
        ":node_id_drsw_on": ["-DNODE_ID_DRSW=1"],
        "//conditions:default": [],
    }) + select({
        ":nid_srdw_hist_on": ["-DNODE_ID_DRSW_SRDW_HIST=1"],
        "//conditions:default": [],
    }) + select({
        ":srdw_label_hist_on": ["-DNODE_ID_DRSW_SRDW_LABEL_HIST=1"],
        "//conditions:default": [],
    }) + select({
        ":symmetric_proj_on": ["-DSYMMETRIC_PROJ=1"],
        "//conditions:default": [],
    }) + select({
        ":offsets_only_on": ["-DOFFSETS_ONLY=1"],
        "//conditions:default": [],
    }) + select({
        ":two_pass_on": ["-DTWO_PASS=1"],
        "//conditions:default": [],
    }) + select({
        ":split_smem_on": ["-DSPLIT_SMEM=1"],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library_ydf(
    name = "label",
    srcs = ["label.cc"],
    hdrs = ["label.h"],
    deps = [
        ":splitter",
        ":uplift",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:distribution",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library_ydf(
    name = "preprocessing",
    srcs = ["preprocessing.cc"],
    hdrs = ["preprocessing.h"],
    deps = [
        ":decision_tree_cc_proto",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

cc_library_ydf(
    name = "splitter",
    srcs = [],
    hdrs = [
        "splitter_accumulator.h",
        "splitter_scanner.h",
    ],
    deps = [
        ":decision_tree_cc_proto",
        ":preprocessing",
        ":uplift",
        ":utils",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:cast",
        "//yggdrasil_decision_forests/utils:circular_buffer",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:distribution",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_highway//:hwy",
        "@com_google_highway//hwy/contrib/sort:vqsort",
        "@com_google_highway//hwy/contrib/sort:vqsort_shared",
        "@eigen",
    ] + select({
        "//conditions:default": [
        ],
    }),
    alwayslink = 1,
)

cc_library_ydf(
    name = "uplift",
    srcs = [],
    hdrs = ["uplift.h"],
    deps = [
        ":decision_tree_cc_proto",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:logging",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library_ydf(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:bitmap",
        "//yggdrasil_decision_forests/utils:concurrency",
        "//yggdrasil_decision_forests/utils:logging",
    ],
)

cc_library_ydf(
    name = "generic_parameters",
    srcs = ["generic_parameters.cc"],
    hdrs = ["generic_parameters.h"],
    deps = [
        ":decision_tree_cc_proto",
        "//yggdrasil_decision_forests/learner:abstract_learner",
        "//yggdrasil_decision_forests/utils:hyper_parameters",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:status_macros",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

# TODO: Enbable CUDA build in OSS

cc_library_ydf(
    name = "gpu",
    srcs = ["gpu.cc"],
    hdrs = ["gpu.h"],
    defines = [] + select({
        # TODO: Add in OSS
        "//conditions:default": ["CUDA_DISABLED"],
    }),
    local_defines = select({
        ":bfs_only_on": ["BFS_ONLY=1"],
        "//conditions:default": [],
    }),
    deps = [
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/utils:status_macros",
        "//yggdrasil_decision_forests/utils:synchronization_primitives",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ] + select({
        # TODO: Add in OSS
        "//conditions:default": [],
    }),
)

# Proto
# ========

all_proto_library(
    name = "decision_tree_proto",
    srcs = ["decision_tree.proto"],
    deps = ["//yggdrasil_decision_forests/utils:distribution_proto"],
)

# Test
# =========================================================

cc_test(
    name = "generic_parameters_test",
    srcs = ["generic_parameters_test.cc"],
    deps = [
        ":generic_parameters",
        "//yggdrasil_decision_forests/utils:test",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "decision_tree_test",
    srcs = ["decision_tree_test.cc"],
    data = [
        "//yggdrasil_decision_forests/test_data",
    ],
    deps = [
        ":decision_tree_cc_proto",
        ":generic_parameters",
        ":label",
        ":preprocessing",
        ":splitter",
        ":training",
        ":uplift",
        ":utils",
        "//yggdrasil_decision_forests/dataset:csv_example_reader",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:data_spec_inference",
        "//yggdrasil_decision_forests/dataset:example_cc_proto",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/dataset:vertical_dataset_io",
        "//yggdrasil_decision_forests/learner:abstract_learner",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/model/decision_tree:decision_tree_cc_proto",
        "//yggdrasil_decision_forests/utils:distribution",
        "//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "//yggdrasil_decision_forests/utils:filesystem",
        "//yggdrasil_decision_forests/utils:hyper_parameters",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:test",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "training_test",
    srcs = ["training_test.cc"],
    deps = [
        ":decision_tree_cc_proto",
        ":gpu",
        ":label",
        ":preprocessing",
        ":training",
        ":utils",
        "//yggdrasil_decision_forests/dataset:data_spec",
        "//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/model:abstract_model_cc_proto",
        "//yggdrasil_decision_forests/model/decision_tree",
        "//yggdrasil_decision_forests/utils:distribution",
        "//yggdrasil_decision_forests/utils:logging",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "preprocessing_test",
    srcs = ["preprocessing_test.cc"],
    deps = [
        ":decision_tree_cc_proto",
        ":preprocessing",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/learner:abstract_learner_cc_proto",
        "//yggdrasil_decision_forests/utils:test",
        "//yggdrasil_decision_forests/utils:testing_macros",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "gpu_test",
    srcs = ["gpu_test.cc"],
    defines = [] + select({
        # TODO: Add in OSS
        "//conditions:default": ["CUDA_DISABLED"],
    }),
    tags = ["cuda_dep"],
    deps = [
        ":gpu",
        "//yggdrasil_decision_forests/dataset:types",
        "//yggdrasil_decision_forests/dataset:vertical_dataset",
        "//yggdrasil_decision_forests/utils:random",
        "//yggdrasil_decision_forests/utils:test",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
