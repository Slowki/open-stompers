use std::path::{Path, PathBuf};

use clap::ValueEnum;
use open_stompers_microduck::MicroduckConfig;
use open_stompers_mini_pi::MiniPiPlusConfig;
use serde::Deserialize;
use thiserror::Error;

const CONFIG_DIRECTORY_NAME: &str = "open-stompers";
const CONFIG_FILE_NAME: &str = "config.json";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Platform {
    MiniPiPlus,
    Microduck,
}

#[derive(Debug, Deserialize)]
pub(super) struct ApplicationConfig {
    pub(super) platform: Option<Platform>,
    #[serde(default)]
    pub(super) mini_pi_plus: MiniPiPlusConfig,
    #[serde(default)]
    pub(super) mini_pi_plus_policy: Option<MiniPiPlusPolicyConfig>,
    #[serde(default)]
    pub(super) microduck: MicroduckConfig,
}

#[derive(Debug, Deserialize)]
pub(super) struct MiniPiPlusPolicyConfig {
    pub(super) model_path: PathBuf,
}

#[derive(Debug, Error)]
pub(super) enum ConfigurationError {
    #[error("could not determine the user configuration directory")]
    ConfigurationDirectoryUnavailable,
    #[error("failed to create default configuration {path}: {source}")]
    CreateDefaultConfiguration {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read configuration {path}: {source}")]
    ReadConfiguration {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse configuration {path}: {source}")]
    ParseConfiguration {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("select a platform with --platform or set `platform` in {0}")]
    PlatformRequired(PathBuf),
}

pub(super) fn default_configuration_path() -> Result<PathBuf, ConfigurationError> {
    let mut path =
        dirs::config_dir().ok_or(ConfigurationError::ConfigurationDirectoryUnavailable)?;
    path.push(CONFIG_DIRECTORY_NAME);
    path.push(CONFIG_FILE_NAME);
    Ok(path)
}

pub(super) fn load_configuration(
    path: &Path,
    is_default: bool,
) -> Result<ApplicationConfig, ConfigurationError> {
    if is_default && !path.exists() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| {
                ConfigurationError::CreateDefaultConfiguration {
                    path: path.to_owned(),
                    source,
                }
            })?;
        }
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
        {
            Ok(mut file) => std::io::Write::write_all(&mut file, b"{}\n").map_err(|source| {
                ConfigurationError::CreateDefaultConfiguration {
                    path: path.to_owned(),
                    source,
                }
            })?,
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(source) => {
                return Err(ConfigurationError::CreateDefaultConfiguration {
                    path: path.to_owned(),
                    source,
                });
            }
        }
    }

    let contents =
        std::fs::read_to_string(path).map_err(|source| ConfigurationError::ReadConfiguration {
            path: path.to_owned(),
            source,
        })?;
    serde_json::from_str(&contents).map_err(|source| ConfigurationError::ParseConfiguration {
        path: path.to_owned(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_configuration_uses_driver_defaults() {
        let directory = tempfile::tempdir().expect("temporary directory should be created");
        let path = directory.path().join("robot.json");
        std::fs::write(
            &path,
            r#"{"platform":"mini-pi-plus","mini_pi_plus":{"imu_serial_path":"/dev/imu"},"mini_pi_plus_policy":{"model_path":"models/actor.onnx"}}"#,
        )
        .expect("configuration should be written");

        let configuration = load_configuration(&path, false).expect("configuration should load");

        assert_eq!(configuration.platform, Some(Platform::MiniPiPlus));
        assert_eq!(configuration.mini_pi_plus.imu_serial_path, "/dev/imu");
        assert_eq!(
            configuration.mini_pi_plus_policy.unwrap().model_path,
            PathBuf::from("models/actor.onnx")
        );
        assert_eq!(
            configuration.mini_pi_plus.motor_serial_paths,
            MiniPiPlusConfig::default().motor_serial_paths
        );
        assert_eq!(
            configuration.mini_pi_plus.motor_baudrate,
            MiniPiPlusConfig::default().motor_baudrate
        );
    }

    #[test]
    fn malformed_configuration_reports_its_path() {
        let directory = tempfile::tempdir().expect("temporary directory should be created");
        let path = directory.path().join("broken.json");
        std::fs::write(&path, "{").expect("configuration should be written");

        let error = load_configuration(&path, false).expect_err("invalid JSON should fail");

        assert!(
            error
                .to_string()
                .contains(path.to_str().expect("path is UTF-8"))
        );
    }

    #[test]
    fn missing_default_configuration_is_created_without_overwriting() {
        let directory = tempfile::tempdir().expect("temporary directory should be created");
        let path = directory.path().join("nested/config.json");

        let configuration = load_configuration(&path, true).expect("default config should load");
        assert!(configuration.platform.is_none());
        assert_eq!(
            std::fs::read_to_string(&path)
                .expect("file should exist")
                .trim(),
            "{}"
        );

        std::fs::write(&path, r#"{"platform":"microduck"}"#)
            .expect("configuration should be replaced for test");
        let configuration = load_configuration(&path, true).expect("existing config should load");
        assert_eq!(configuration.platform, Some(Platform::Microduck));
    }

    #[test]
    fn partial_microduck_configuration_uses_driver_defaults() {
        let directory = tempfile::tempdir().expect("temporary directory should be created");
        let path = directory.path().join("robot.json");
        std::fs::write(
            &path,
            r#"{"platform":"microduck","microduck":{"serial_path":"/dev/ttyS9","imu_mount":"beta"}}"#,
        )
        .expect("configuration should be written");

        let configuration = load_configuration(&path, false).expect("configuration should load");

        assert_eq!(configuration.microduck.serial_path, "/dev/ttyS9");
        assert_eq!(
            configuration.microduck.imu_mount,
            open_stompers_microduck::ImuMount::Beta
        );
        assert_eq!(
            configuration.microduck.baudrate,
            MicroduckConfig::default().baudrate
        );
        assert!(configuration.microduck.fast_sync_read);
    }
}
