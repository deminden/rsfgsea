pub use crate::RSampleKind;
#[cfg(feature = "gpu")]
pub use crate::algo::run_gsea_gpu_with_config;
pub use crate::algo::{
    calculate_es_fgsea, fgsea, fgsea_multilevel_with_sample_size,
    fgsea_multilevel_with_sample_size_and_kind, fgsea_simple_with_sample_size,
    fgsea_simple_with_sample_size_and_kind, fgsea_with_sample_size,
    fgsea_with_sample_size_and_kind,
};
pub use crate::bindings::{
    ExecutionPlan, InterfaceMode, parse_interface_mode, resolve_execution_plan,
};
pub use crate::blitz::fgsea_blitz_with_options;
pub use crate::core::{
    BlitzOptions, DecorCacheMode, DecorCorrelation, DecorOptions, DecorPreset,
    DecorPresetResolution, DecorRedundancy, DecorStringencyResolution, DecorWeightFormula,
    EnrichmentMethod, EnrichmentResult, EnrichmentResultExport, Pathway, PathwayDb, RankedList,
    ScoreType, resolve_decor_preset, resolve_decor_stringency,
};
pub use crate::decor::{
    DecorCache, DecorCacheMetadata, DecorCacheStatus, calculate_es_decor,
    ensure_decor_cache_for_paths, fgsea_decor_simple_with_options,
    fgsea_decor_simple_with_sample_size,
};
pub use crate::io::{read_gmt, read_ranked_list};
pub use crate::plot::{
    EnrichmentPlotData, EnrichmentPlotOptions, GseaTablePlotData, GseaTablePlotOptions,
    build_enrichment_plot_data, build_gsea_table_plot_data, write_enrichment_plot_png,
    write_gsea_table_plot_png,
};

#[cfg(feature = "gpu")]
pub use crate::gpu::GpuEngine;
