//! Skill registry for storing and querying skills.

use std::collections::HashMap;

use crate::{Skill, SkillError, SkillLoader};

/// Registry of available skills.
///
/// Stores skills by name for explicit lookup and loading.
///
/// # Example
///
/// ```rust,ignore
/// let loader = SkillLoader::new().add_path("~/.aither/skills");
/// let mut registry = SkillRegistry::new();
/// registry.load_from(&loader).await?;
///
/// assert!(registry.get("code-review").is_none());
/// ```
#[derive(Debug, Default)]
pub struct SkillRegistry {
    skills: HashMap<String, Skill>,
}

impl SkillRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Load skills from a loader.
    ///
    /// Skills are added to the registry, potentially overwriting
    /// existing skills with the same name.
    ///
    /// # Errors
    ///
    /// Returns an error if the loader fails to load skills.
    pub async fn load_from(&mut self, loader: &SkillLoader) -> Result<usize, SkillError> {
        let skills = loader.load_all().await?;
        let count = skills.len();

        for skill in skills {
            self.skills.insert(skill.name.clone(), skill);
        }

        Ok(count)
    }

    /// Register a skill directly.
    pub fn register(&mut self, skill: Skill) {
        self.skills.insert(skill.name.clone(), skill);
    }

    /// Get a skill by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    /// Check if a skill exists.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.skills.contains_key(name)
    }

    /// Get all registered skill names.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.skills.keys().map(String::as_str).collect()
    }

    /// Get all registered skills.
    #[must_use]
    pub fn all(&self) -> Vec<&Skill> {
        self.skills.values().collect()
    }

    /// Number of registered skills.
    #[must_use]
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Check if registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Remove a skill from the registry.
    pub fn remove(&mut self, name: &str) -> Option<Skill> {
        self.skills.remove(name)
    }

    /// Clear all skills from the registry.
    pub fn clear(&mut self) {
        self.skills.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_skill(name: &str) -> Skill {
        Skill {
            name: name.to_string(),
            description: format!("{name} description"),
            instructions: format!("Instructions for {name}"),
            allowed_tools: None,
            resources: HashMap::new(),
        }
    }

    #[test]
    fn test_register_and_get() {
        let mut registry = SkillRegistry::new();
        registry.register(make_skill("test"));

        assert!(registry.contains("test"));
        assert_eq!(
            registry.get("test").map(|s| &s.name),
            Some(&"test".to_string())
        );
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_names_and_all() {
        let mut registry = SkillRegistry::new();
        registry.register(make_skill("alpha"));
        registry.register(make_skill("beta"));

        let names = registry.names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));

        assert_eq!(registry.all().len(), 2);
    }

    #[test]
    fn test_remove_and_clear() {
        let mut registry = SkillRegistry::new();
        registry.register(make_skill("a"));
        registry.register(make_skill("b"));

        assert_eq!(registry.len(), 2);

        registry.remove("a");
        assert_eq!(registry.len(), 1);
        assert!(!registry.contains("a"));

        registry.clear();
        assert!(registry.is_empty());
    }
}
