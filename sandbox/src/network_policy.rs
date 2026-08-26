//! Declarative host-based network policy.
//!
//! Applications describe which hosts sandboxed commands may reach as a
//! serializable [`NetworkPolicyConfig`]; compiling it yields a
//! [`CompiledNetworkPolicy`] whose [`allows_domain`](CompiledNetworkPolicy::allows_domain)
//! verdicts back a [`PermissionHandler::check_domain`](crate::PermissionHandler::check_domain)
//! implementation.

use globset::{Glob, GlobSet, GlobSetBuilder};

/// Global network restriction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NetworkPolicyMode {
    /// No restrictions: every domain is allowed.
    #[default]
    Disabled,
    /// Deny all network access.
    Restricted,
    /// Allow only hosts matching the configured patterns.
    AllowList,
    /// Allow everything except hosts matching the configured patterns.
    DenyList,
}

impl NetworkPolicyMode {
    /// Stable `snake_case` identifier for persistence and telemetry.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Restricted => "restricted",
            Self::AllowList => "allow_list",
            Self::DenyList => "deny_list",
        }
    }
}

/// Error compiling a [`NetworkPolicyConfig`].
#[derive(Debug, thiserror::Error)]
pub enum NetworkPolicyError {
    /// A configured host entry is not a valid glob pattern.
    #[error("invalid network host pattern '{pattern}': {source}")]
    InvalidPattern {
        /// The offending host pattern as configured.
        pattern: String,
        /// The underlying glob parse error.
        source: globset::Error,
    },
    /// The combined host matcher could not be built.
    #[error("failed to build host matcher: {0}")]
    Matcher(#[source] globset::Error),
}

/// Persisted network policy settings.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct NetworkPolicyConfig {
    /// Global restriction mode.
    #[serde(default)]
    pub mode: NetworkPolicyMode,
    /// Host patterns interpreted according to [`NetworkPolicyMode`].
    #[serde(default)]
    pub hosts: Vec<String>,
}

impl NetworkPolicyConfig {
    /// Whether the policy restricts anything at all.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        !matches!(self.mode, NetworkPolicyMode::Disabled)
    }

    /// Compile the settings into a matchable policy.
    ///
    /// Returns `Ok(None)` when the policy is disabled.
    ///
    /// # Errors
    /// Returns an error when a configured host pattern is not a valid glob
    /// or the combined matcher cannot be built.
    pub fn compile(&self) -> Result<Option<CompiledNetworkPolicy>, NetworkPolicyError> {
        if !self.is_active() {
            return Ok(None);
        }

        let hosts = self
            .hosts
            .iter()
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>();

        let matcher = if matches!(
            self.mode,
            NetworkPolicyMode::AllowList | NetworkPolicyMode::DenyList
        ) {
            Some(build_matcher(&hosts)?)
        } else {
            None
        };

        Ok(Some(CompiledNetworkPolicy {
            mode: self.mode,
            hosts,
            matcher,
        }))
    }
}

/// A compiled, matchable network policy.
#[derive(Debug, Clone)]
pub struct CompiledNetworkPolicy {
    mode: NetworkPolicyMode,
    hosts: Vec<String>,
    matcher: Option<GlobSet>,
}

impl CompiledNetworkPolicy {
    /// The mode this policy was compiled from.
    #[must_use]
    pub const fn mode(&self) -> NetworkPolicyMode {
        self.mode
    }

    /// Normalized host patterns this policy matches against.
    #[must_use]
    pub fn hosts(&self) -> &[String] {
        &self.hosts
    }

    /// Whether the policy permits connecting to `domain`.
    #[must_use]
    pub fn allows_domain(&self, domain: &str) -> bool {
        let normalized = domain.trim().to_ascii_lowercase();
        match self.mode {
            NetworkPolicyMode::Disabled => true,
            NetworkPolicyMode::Restricted => false,
            NetworkPolicyMode::AllowList => self
                .matcher
                .as_ref()
                .is_some_and(|matcher| matcher.is_match(normalized.as_str())),
            NetworkPolicyMode::DenyList => !self
                .matcher
                .as_ref()
                .is_some_and(|matcher| matcher.is_match(normalized.as_str())),
        }
    }
}

fn build_matcher(hosts: &[String]) -> Result<GlobSet, NetworkPolicyError> {
    let mut builder = GlobSetBuilder::new();
    for host in hosts {
        let glob = Glob::new(host).map_err(|source| NetworkPolicyError::InvalidPattern {
            pattern: host.clone(),
            source,
        })?;
        builder.add(glob);
    }
    builder.build().map_err(NetworkPolicyError::Matcher)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restricted_policy_blocks_everything() {
        let policy = NetworkPolicyConfig {
            mode: NetworkPolicyMode::Restricted,
            hosts: Vec::new(),
        }
        .compile()
        .expect("restricted policy should compile")
        .expect("restricted policy should be active");
        assert!(!policy.allows_domain("example.com"));
    }

    #[test]
    fn allow_list_policy_matches_globs() {
        let policy = NetworkPolicyConfig {
            mode: NetworkPolicyMode::AllowList,
            hosts: vec!["*.example.com".to_string()],
        }
        .compile()
        .expect("allow list should compile")
        .expect("allow list should be active");
        assert!(policy.allows_domain("api.example.com"));
        assert!(!policy.allows_domain("example.org"));
    }

    #[test]
    fn deny_list_policy_blocks_matches_only() {
        let policy = NetworkPolicyConfig {
            mode: NetworkPolicyMode::DenyList,
            hosts: vec!["*.tracker.net".to_string()],
        }
        .compile()
        .expect("deny list should compile")
        .expect("deny list should be active");
        assert!(!policy.allows_domain("ads.tracker.net"));
        assert!(policy.allows_domain("api.github.com"));
    }

    #[test]
    fn disabled_policy_compiles_to_none() {
        let compiled = NetworkPolicyConfig::default()
            .compile()
            .expect("disabled policy should compile");
        assert!(compiled.is_none());
    }

    #[test]
    fn invalid_pattern_reports_offending_host() {
        let error = NetworkPolicyConfig {
            mode: NetworkPolicyMode::AllowList,
            hosts: vec!["[".to_string()],
        }
        .compile()
        .expect_err("invalid glob must fail compilation");
        assert!(error.to_string().contains('['));
    }
}
