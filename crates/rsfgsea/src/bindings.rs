#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceMode {
    Multilevel,
    Simple,
    Fgsea,
    Blitz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPlan {
    Cpu(InterfaceMode),
    Gpu {
        mode: InterfaceMode,
        n_perm: usize,
        allow_multilevel: bool,
    },
}

pub fn parse_interface_mode(mode: &str) -> Result<InterfaceMode, String> {
    match mode.to_lowercase().as_str() {
        "multilevel" => Ok(InterfaceMode::Multilevel),
        "simple" => Ok(InterfaceMode::Simple),
        "fgsea" => Ok(InterfaceMode::Fgsea),
        "blitz" => Ok(InterfaceMode::Blitz),
        other => Err(format!(
            "Invalid mode '{}'. Expected one of: fgsea, multilevel, simple, blitz.",
            other
        )),
    }
}

pub fn resolve_execution_plan(
    mode: InterfaceMode,
    gpu: bool,
    nperm: Option<usize>,
    n_perm_simple: usize,
) -> Result<ExecutionPlan, String> {
    if mode == InterfaceMode::Blitz && nperm.is_some() {
        return Err("nperm is not supported with mode='blitz'.".to_string());
    }

    if !gpu {
        return Ok(ExecutionPlan::Cpu(mode));
    }

    if mode == InterfaceMode::Blitz {
        return Err("gpu is not supported with mode='blitz'.".to_string());
    }

    if mode == InterfaceMode::Multilevel && nperm.is_some() {
        return Err("nperm is only valid with mode='fgsea' or mode='simple'.".to_string());
    }

    let (n_perm, allow_multilevel) = match mode {
        InterfaceMode::Multilevel => (n_perm_simple, true),
        InterfaceMode::Simple => (nperm.unwrap_or(n_perm_simple), false),
        InterfaceMode::Fgsea => (nperm.unwrap_or(n_perm_simple), nperm.is_none()),
        InterfaceMode::Blitz => unreachable!("blitz mode is CPU-only and handled above"),
    };

    Ok(ExecutionPlan::Gpu {
        mode,
        n_perm,
        allow_multilevel,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_interface_mode_accepts_supported_values() {
        assert_eq!(parse_interface_mode("fgsea").unwrap(), InterfaceMode::Fgsea);
        assert_eq!(
            parse_interface_mode("multilevel").unwrap(),
            InterfaceMode::Multilevel
        );
        assert_eq!(
            parse_interface_mode("simple").unwrap(),
            InterfaceMode::Simple
        );
        assert_eq!(parse_interface_mode("blitz").unwrap(), InterfaceMode::Blitz);
    }

    #[test]
    fn resolve_execution_plan_for_cpu_preserves_requested_mode() {
        assert_eq!(
            resolve_execution_plan(InterfaceMode::Fgsea, false, None, 1000).unwrap(),
            ExecutionPlan::Cpu(InterfaceMode::Fgsea)
        );
        assert_eq!(
            resolve_execution_plan(InterfaceMode::Multilevel, false, None, 1000).unwrap(),
            ExecutionPlan::Cpu(InterfaceMode::Multilevel)
        );
        assert_eq!(
            resolve_execution_plan(InterfaceMode::Simple, false, Some(250), 1000).unwrap(),
            ExecutionPlan::Cpu(InterfaceMode::Simple)
        );
        assert_eq!(
            resolve_execution_plan(InterfaceMode::Blitz, false, None, 1000).unwrap(),
            ExecutionPlan::Cpu(InterfaceMode::Blitz)
        );
    }

    #[test]
    fn resolve_execution_plan_for_gpu_exposes_wrapper_and_explicit_modes() {
        assert_eq!(
            resolve_execution_plan(InterfaceMode::Fgsea, true, None, 1000).unwrap(),
            ExecutionPlan::Gpu {
                mode: InterfaceMode::Fgsea,
                n_perm: 1000,
                allow_multilevel: true,
            }
        );
        assert_eq!(
            resolve_execution_plan(InterfaceMode::Fgsea, true, Some(250), 1000).unwrap(),
            ExecutionPlan::Gpu {
                mode: InterfaceMode::Fgsea,
                n_perm: 250,
                allow_multilevel: false,
            }
        );
        assert_eq!(
            resolve_execution_plan(InterfaceMode::Simple, true, Some(250), 1000).unwrap(),
            ExecutionPlan::Gpu {
                mode: InterfaceMode::Simple,
                n_perm: 250,
                allow_multilevel: false,
            }
        );
        assert_eq!(
            resolve_execution_plan(InterfaceMode::Multilevel, true, None, 1000).unwrap(),
            ExecutionPlan::Gpu {
                mode: InterfaceMode::Multilevel,
                n_perm: 1000,
                allow_multilevel: true,
            }
        );
    }

    #[test]
    fn resolve_execution_plan_rejects_nperm_for_multilevel_gpu() {
        let err =
            resolve_execution_plan(InterfaceMode::Multilevel, true, Some(250), 1000).unwrap_err();
        assert_eq!(
            err,
            "nperm is only valid with mode='fgsea' or mode='simple'."
        );
    }

    #[test]
    fn resolve_execution_plan_rejects_incompatible_blitz_args() {
        assert_eq!(
            resolve_execution_plan(InterfaceMode::Blitz, true, None, 1000).unwrap_err(),
            "gpu is not supported with mode='blitz'."
        );
        assert_eq!(
            resolve_execution_plan(InterfaceMode::Blitz, false, Some(100), 1000).unwrap_err(),
            "nperm is not supported with mode='blitz'."
        );
    }
}
