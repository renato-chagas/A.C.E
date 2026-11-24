//! Launch an installed Windows backend without depending on a checkout or package manager.

use std::path::{Path, PathBuf};

const JARVIS_ENTRY_POINT: &str = "from openjarvis.cli import main; main()";
const VERIFY_BACKEND: &str = include_str!("offline_backend.py");

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) struct AttachedBackend {
    pub(crate) pid: u32,
    pub(crate) started_at: String,
    pub(crate) mode: AttachedBackendMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) cloudflare_model: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AttachedBackendMode {
    LocalServices,
    Inference,
    Cloudflare,
}

pub(crate) struct BundledBackend {
    root: PathBuf,
    python: PathBuf,
}

impl BundledBackend {
    pub(crate) fn open(root: &Path) -> Result<Self, String> {
        let python = root.join("python.exe");
        if !python.is_file() {
            return Err(
                "The packaged Python runtime is missing. Reinstall the ACE offline package.".into(),
            );
        }
        Ok(Self {
            root: root.to_owned(),
            python,
        })
    }

    pub(crate) fn command(&self) -> tokio::process::Command {
        let mut cmd = self.python_command();
        cmd.args(["-I", "-B", "-c", JARVIS_ENTRY_POINT, "--quiet"]);
        cmd
    }

    fn python_command(&self) -> tokio::process::Command {
        let mut cmd = super::local_command(&self.python);
        cmd.current_dir(&self.root)
            .env("OPENJARVIS_HOME", profile_data_dir());
        if std::env::var_os("OPENJARVIS_CONFIG").is_some_and(|value| !value.is_empty()) {
            cmd.env("OPENJARVIS_CONFIG", profile_config_path());
        }
        for key in super::MANAGED_CLOUD_KEY_NAMES {
            cmd.env_remove(key);
        }
        #[cfg(target_os = "windows")]
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        cmd
    }

    async fn verify_backend(
        &self,
        port: u16,
        stop: Option<&AttachedBackend>,
    ) -> Result<std::process::Output, String> {
        let mut cmd = self.python_command();
        cmd.args(["-I", "-B", "-c", VERIFY_BACKEND])
            .arg(port.to_string())
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true);
        if let Some(stop) = stop {
            cmd.arg(serde_json::to_string(stop).map_err(|error| error.to_string())?);
        }
        tokio::time::timeout(std::time::Duration::from_secs(12), cmd.output())
            .await
            .map_err(|_| "The existing ACE backend could not be verified in time.".to_owned())?
            .map_err(|_| "The packaged ACE backend verifier could not run.".to_owned())
    }

    pub(crate) async fn identify_server(&self, port: u16) -> Option<AttachedBackend> {
        let output = self.verify_backend(port, None).await.ok()?;
        if !output.status.success() {
            return None;
        }
        serde_json::from_slice(&output.stdout).ok().flatten()
    }

    pub(crate) async fn stop_attached_server(
        &self,
        port: u16,
        target: &AttachedBackend,
    ) -> Result<(), String> {
        let output = self.verify_backend(port, Some(target)).await?;
        if !output.status.success() {
            return Err("The attached ACE backend changed or could not be stopped.".into());
        }
        Ok(())
    }

    pub(crate) async fn initialize_profile(&self) -> Result<(), String> {
        if std::env::var_os("OPENJARVIS_CONFIG").is_some_and(|value| !value.is_empty())
            || profile_config_path().is_file()
        {
            return Ok(());
        }
        let mut cmd = self.command();
        let output = cmd
            .args(["init", "--preset", "ace", "--no-analytics", "--no-download"])
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true)
            .output()
            .await
            .map_err(|err| format!("Could not initialize the local ACE profile: {err}"))?;
        if output.status.success() && profile_config_path().is_file() {
            return Ok(());
        }
        Err("Could not initialize the local ACE profile from the packaged preset.".into())
    }
}

/// Match OpenJarvis' environment-aware configuration location.
pub(crate) fn profile_config_path() -> PathBuf {
    if let Some(path) = std::env::var_os("OPENJARVIS_CONFIG").filter(|v| !v.is_empty()) {
        return absolute_profile_path(PathBuf::from(path));
    }
    profile_data_dir().join("config.toml")
}

pub(crate) fn profile_data_dir() -> PathBuf {
    let home = std::env::var_os("OPENJARVIS_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_DATA_HOME")
                .filter(|v| !v.is_empty())
                .map(|path| PathBuf::from(path).join("openjarvis"))
        })
        .unwrap_or_else(|| profile_home().join(".openjarvis"));
    absolute_profile_path(home)
}

fn profile_home() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(super::home_dir()))
}

fn absolute_profile_path(path: PathBuf) -> PathBuf {
    let path = match path.strip_prefix("~") {
        Ok(rest) => profile_home().join(rest),
        Err(_) => path,
    };
    if path.is_absolute() {
        path
    } else {
        std::env::current_dir().map_or(path.clone(), |cwd| cwd.join(path))
    }
}

pub(crate) fn is_loopback_url(value: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(value) else {
        return false;
    };
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port() == Some(0)
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return false;
    }
    let hostname = url.host_str().unwrap_or_default();
    hostname.eq_ignore_ascii_case("localhost")
        || hostname
            .trim_start_matches('[')
            .trim_end_matches(']')
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

#[cfg(test)]
mod tests {
    use super::{is_loopback_url, BundledBackend};

    #[test]
    fn offline_endpoints_are_strictly_loopback() {
        for allowed in [
            "http://localhost:1234",
            "http://127.0.0.1:11434",
            "http://[::1]:1234",
        ] {
            assert!(is_loopback_url(allowed), "{allowed}");
        }
        for blocked in [
            "http://localhost.example:1234",
            "http://localhost@192.0.2.1",
            "http://user@localhost",
            "http://127.0.0.1:0",
            "http://localhost:1234/?token=test",
            "http://localhost:1234/#section",
            "file://localhost/path",
            "http://0.0.0.0:8000",
        ] {
            assert!(!is_loopback_url(blocked), "{blocked}");
        }
    }

    #[test]
    fn offline_runtime_does_not_fall_back_to_path_when_missing() {
        assert!(BundledBackend::open(std::path::Path::new("no-such-ace-resource-dir")).is_err());
    }
}
