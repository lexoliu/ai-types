//! Typed container runtime specifications shared by host integrations.
//!
//! This module defines type-safe container image, mount, and launch specs so
//! application code can construct container requests without relying on
//! positional arguments or ad-hoc string conventions.

use std::path::{Component, Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Supported Docker-compatible runtime kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ContainerRuntimeKind {
    /// `OrbStack`, a macOS-native Docker-compatible runtime.
    OrbStack,
    /// Podman, the daemonless Docker-compatible runtime.
    Podman,
    /// Docker Engine itself.
    Docker,
}

impl std::fmt::Display for ContainerRuntimeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OrbStack => write!(f, "OrbStack"),
            Self::Podman => write!(f, "Podman"),
            Self::Docker => write!(f, "Docker"),
        }
    }
}

/// Runtime selection order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RuntimePreference {
    /// Runtimes to try, most preferred first.
    pub order: Vec<ContainerRuntimeKind>,
}

impl Default for RuntimePreference {
    fn default() -> Self {
        Self {
            order: vec![
                ContainerRuntimeKind::OrbStack,
                ContainerRuntimeKind::Podman,
                ContainerRuntimeKind::Docker,
            ],
        }
    }
}

/// Filesystem mount access policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MountAccess {
    /// The container may read the mount but not write to it.
    ReadOnly,
    /// The container may read and write the mount.
    ReadWrite,
}

impl MountAccess {
    /// Whether this policy forbids writes.
    #[must_use]
    pub const fn read_only(self) -> bool {
        matches!(self, Self::ReadOnly)
    }
}

/// A host path mounted into a container path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MountSpec {
    /// Path on the host to expose.
    pub host_path: PathBuf,
    /// Path inside the container where it appears.
    pub container_path: PathBuf,
    /// Whether the container may write to it.
    pub access: MountAccess,
}

/// Declarative host/container root pair used to create validated child mounts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MountRoot {
    /// Base directory on the host.
    pub host_root: PathBuf,
    /// Directory the host root maps to inside the container.
    pub container_root: PathBuf,
}

/// Mount-root composition error.
#[derive(Debug, thiserror::Error)]
pub enum MountRootError {
    /// The relative path was empty, which would mount the root itself.
    #[error("relative path must not be empty")]
    /// Relative path was empty.
    EmptyRelativePath,
    /// An absolute path was given where a path relative to the root was expected.
    #[error("relative path must not be absolute: {0}")]
    /// Relative path was absolute.
    AbsoluteRelativePath(PathBuf),
    /// The path contained `..`, which would escape the mount root.
    #[error("relative path must not contain parent traversal: {0}")]
    /// Relative path contained parent traversal.
    ParentTraversal(PathBuf),
}

impl MountRoot {
    /// Pairs a host directory with the container path it maps to.
    #[must_use]
    pub fn new(host_root: impl Into<PathBuf>, container_root: impl Into<PathBuf>) -> Self {
        Self {
            host_root: host_root.into(),
            container_root: container_root.into(),
        }
    }

    /// Derives a mount for a path beneath this root.
    ///
    /// # Errors
    ///
    /// Returns [`MountRootError`] if `relative` is empty, absolute, or contains
    /// a `..` component — any of which would place the mount outside the root.
    pub fn child(
        &self,
        relative: impl AsRef<Path>,
        access: MountAccess,
    ) -> Result<MountSpec, MountRootError> {
        let relative = relative.as_ref();
        Self::validate_relative_path(relative)?;

        let host_path = self.host_root.join(relative);
        let container_path = self.container_root.join(relative);
        Ok(match access {
            MountAccess::ReadOnly => MountSpec::read_only(host_path, container_path),
            MountAccess::ReadWrite => MountSpec::read_write(host_path, container_path),
        })
    }

    /// Derives a read-only mount beneath this root.
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::child`].
    pub fn read_only_child(&self, relative: impl AsRef<Path>) -> Result<MountSpec, MountRootError> {
        self.child(relative, MountAccess::ReadOnly)
    }

    /// Derives a writable mount beneath this root.
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::child`].
    pub fn read_write_child(
        &self,
        relative: impl AsRef<Path>,
    ) -> Result<MountSpec, MountRootError> {
        self.child(relative, MountAccess::ReadWrite)
    }

    fn validate_relative_path(path: &Path) -> Result<(), MountRootError> {
        if path.as_os_str().is_empty() {
            return Err(MountRootError::EmptyRelativePath);
        }
        if path.is_absolute() {
            return Err(MountRootError::AbsoluteRelativePath(path.to_path_buf()));
        }
        if path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
        {
            return Err(MountRootError::ParentTraversal(path.to_path_buf()));
        }
        Ok(())
    }
}

impl MountSpec {
    /// Mounts a host path into the container without write access.
    #[must_use]
    pub fn read_only(host_path: impl Into<PathBuf>, container_path: impl Into<PathBuf>) -> Self {
        Self {
            host_path: host_path.into(),
            container_path: container_path.into(),
            access: MountAccess::ReadOnly,
        }
    }

    /// Mounts a host path into the container with write access.
    #[must_use]
    pub fn read_write(host_path: impl Into<PathBuf>, container_path: impl Into<PathBuf>) -> Self {
        Self {
            host_path: host_path.into(),
            container_path: container_path.into(),
            access: MountAccess::ReadWrite,
        }
    }
}

/// Image build specification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ContainerImageSpec {
    /// Tag to apply to the built image.
    pub tag: String,
    /// Directory used as the build context.
    pub context_dir: PathBuf,
    /// Dockerfile to build, if not the default one in the context directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dockerfile: Option<PathBuf>,
}

impl ContainerImageSpec {
    /// Describes an image built from `context_dir` and tagged `tag`.
    #[must_use]
    pub fn new(tag: impl Into<String>, context_dir: impl Into<PathBuf>) -> Self {
        Self {
            tag: tag.into(),
            context_dir: context_dir.into(),
            dockerfile: None,
        }
    }

    /// Builds from a specific Dockerfile rather than the context default.
    #[must_use]
    pub fn with_dockerfile(mut self, dockerfile: impl Into<PathBuf>) -> Self {
        self.dockerfile = Some(dockerfile.into());
        self
    }
}

/// Container launch specification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ContainerLaunchSpec {
    /// Name to give the running container.
    pub name: String,
    /// Image to run.
    pub image: String,
    /// Directory the container treats as its workspace.
    pub workspace: PathBuf,
    /// Additional host paths to expose.
    #[serde(default)]
    pub mounts: Vec<MountSpec>,
    /// Environment entries, each formatted as `KEY=value`.
    #[serde(default)]
    pub env: Vec<String>,
}

impl ContainerLaunchSpec {
    /// Describes a container to launch from `image` in `workspace`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        image: impl Into<String>,
        workspace: impl Into<PathBuf>,
    ) -> Self {
        Self {
            name: name.into(),
            image: image.into(),
            workspace: workspace.into(),
            mounts: Vec::new(),
            env: Vec::new(),
        }
    }

    /// Exposes an additional host path inside the container.
    #[must_use]
    pub fn with_mount(mut self, mount: MountSpec) -> Self {
        self.mounts.push(mount);
        self
    }

    /// Adds an environment entry, formatted as `KEY=value`.
    #[must_use]
    pub fn with_env(mut self, value: impl Into<String>) -> Self {
        self.env.push(value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{MountRoot, MountRootError};

    #[test]
    fn mount_root_maps_relative_path() {
        let root = MountRoot::new("/host/.may", "/root/.may");
        let mount = root
            .read_only_child("skills")
            .expect("relative child mount should be valid");
        assert_eq!(
            mount.host_path,
            std::path::PathBuf::from("/host/.may/skills")
        );
        assert_eq!(
            mount.container_path,
            std::path::PathBuf::from("/root/.may/skills")
        );
    }

    #[test]
    fn mount_root_rejects_parent_traversal() {
        let root = MountRoot::new("/host/.may", "/root/.may");
        let error = root
            .read_write_child("../secrets")
            .expect_err("parent traversal must fail");
        assert!(matches!(error, MountRootError::ParentTraversal(_)));
    }
}
