use std::sync::Arc;
use std::time::Duration;
use tauri::menu::{MenuBuilder, MenuItemBuilder};
use tauri::tray::TrayIconBuilder;
use tauri::Manager;
use tauri_plugin_autostart::MacosLauncher;
use tokio::sync::Mutex;

mod cloudflare_credentials;
mod offline;
#[cfg(target_os = "windows")]
mod windows_tcp;

const OLLAMA_PORT: u16 = 11434;
const JARVIS_PORT: u16 = 8000;
const CLOUDFLARE_MODEL: &str = "@cf/google/gemma-4-26b-a4b-it";
const CLOUDFLARE_ENV: [&str; 2] = ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"];
const CLOUDFLARE_INFERENCE_HEADER: &str = "x-ace-inference-source";
type CloudflareEnvironment = [(&'static str, String); 2];

fn local_std_command(program: impl AsRef<std::ffi::OsStr>) -> std::process::Command {
    let mut cmd = std::process::Command::new(program);
    for name in CLOUDFLARE_ENV {
        cmd.env_remove(name);
    }
    cmd
}

fn local_command(program: impl AsRef<std::ffi::OsStr>) -> tokio::process::Command {
    local_std_command(program).into()
}

#[cfg(target_os = "windows")]
fn capture_cloudflare_launch_environment() -> &'static Result<CloudflareEnvironment, String> {
    static LAUNCH_ENVIRONMENT: std::sync::OnceLock<Result<CloudflareEnvironment, String>> =
        std::sync::OnceLock::new();
    LAUNCH_ENVIRONMENT.get_or_init(|| {
        let credentials = read_cloudflare_launch_environment();
        // Windows supports process environment mutation. Capture before Tauri
        // can create WebView or plugin children, which inherit it implicitly.
        for name in CLOUDFLARE_ENV {
            std::env::remove_var(name);
        }
        credentials
    })
}

fn cloudflare_launch_environment() -> Result<CloudflareEnvironment, String> {
    #[cfg(target_os = "windows")]
    return capture_cloudflare_launch_environment().clone();
    #[cfg(not(target_os = "windows"))]
    read_cloudflare_launch_environment()
}

fn read_cloudflare_launch_environment() -> Result<CloudflareEnvironment, String> {
    let [account, token] = CLOUDFLARE_ENV.map(|name| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(|value| (name, value))
    });
    match (account, token) {
        (Some(account), Some(token)) => Ok([account, token]),
        _ => Err("Cloudflare credentials were not provided in the A.C.E. launch environment.".into()),
    }
}

fn configure_cloudflare_environment(
    cmd: &mut tokio::process::Command,
    credentials: &CloudflareEnvironment,
) {
    for name in MANAGED_CLOUD_KEY_NAMES {
        cmd.env_remove(name);
    }
    for (name, value) in credentials {
        cmd.env(name, value);
    }
}

fn cloudflare_response(headers: &reqwest::header::HeaderMap) -> bool {
    headers
        .get_all(CLOUDFLARE_INFERENCE_HEADER)
        .iter()
        .any(|value| value == "cloudflare")
}
#[cfg(not(all(target_os = "windows", target_arch = "aarch64")))]
const DESKTOP_UV_SYNC_COMMAND: &str =
    "uv sync --extra desktop --extra inference-cloud --extra inference-google --group desktop-native";
#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
const DESKTOP_UV_SYNC_COMMAND: &str =
    "uv sync --python cpython-3.12-windows-x86_64-none --extra desktop --extra inference-cloud --extra inference-google --group desktop-native";

/// A resolved boot plan derived purely from the inference config.
/// Pure and side-effect-free so it can be unit-tested without spawning
/// processes or touching the network.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BootPlan {
    /// Whether to start and wait for the bundled Ollama.
    launch_ollama: bool,
    /// The user-selected Ollama model (None for custom endpoints).
    model_to_pull: Option<String>,
    /// Optional `(engine_key, bare_host)` override for a custom endpoint,
    /// e.g. `("lmstudio", "http://localhost:1234")`. Written into
    /// ~/.openjarvis/config.toml so `jarvis serve` picks it up.
    engine_host: Option<(String, String)>,
    /// Args appended after `uv run jarvis serve --port <port>`.
    serve_args: Vec<String>,
}

/// Default OpenAI-compatible engine key used when a custom endpoint config
/// omits one (LM Studio is the canonical local server).
const CUSTOM_FALLBACK_ENGINE: &str = "lmstudio";

/// Decide what to launch/pull/serve from the inference config.
/// Pure: no I/O, no spawning.
fn boot_plan(cfg: &InferenceConfig) -> Result<BootPlan, String> {
    match cfg.kind {
        SourceKind::Ollama => {
            let model =
                normalize_ollama_model_tag(cfg.model.as_deref().unwrap_or("")).map_err(|_| {
                    "Choose an Ollama model tag in Settings before starting inference. \
                     Only that exact model will be used or downloaded."
                        .to_string()
                })?;
            Ok(BootPlan {
                launch_ollama: true,
                model_to_pull: Some(model.clone()),
                engine_host: None,
                serve_args: vec![
                    "--engine".into(),
                    "ollama".into(),
                    "--model".into(),
                    model,
                    "--agent".into(),
                    "simple".into(),
                ],
            })
        }
        SourceKind::Cloudflare => {
            let model = validate_cloudflare_source(
                cfg.model.as_deref(),
                cfg.host.is_some(),
                cfg.engine.is_some(),
                false,
            )?;
            Ok(BootPlan {
                launch_ollama: false,
                model_to_pull: None,
                engine_host: None,
                serve_args: ["--engine", "cloudflare", "--model", model, "--agent", "simple"]
                    .into_iter().map(str::to_owned).collect(),
            })
        }
        SourceKind::Custom => {
            if selects_cloudflare_engine(cfg.engine.as_deref()) {
                return Err("Select the Cloudflare inference source to use Workers AI.".into());
            }
            let engine = cfg
                .engine
                .clone()
                .unwrap_or_else(|| CUSTOM_FALLBACK_ENGINE.to_string());
            // Record (engine_key, bare_host) only when a host is configured, so
            // boot can write `[engine.<key>] host = ...` into config.toml. An
            // empty host is dropped (no override).
            let engine_host = cfg
                .host
                .clone()
                .filter(|h| !h.is_empty())
                .map(|h| (engine.clone(), h));
            // `model` may be empty if the config is malformed; `jarvis serve`
            // surfaces a clear error then (there is no universal default model
            // for an arbitrary endpoint).
            let model = cfg.model.clone().unwrap_or_default();
            Ok(BootPlan {
                launch_ollama: false,
                model_to_pull: None,
                engine_host,
                serve_args: vec![
                    "--engine".into(),
                    engine,
                    "--model".into(),
                    model,
                    "--agent".into(),
                    "simple".into(),
                ],
            })
        }
    }
}

/// Get the user home directory, handling both Unix (HOME) and Windows (USERPROFILE).
fn home_dir() -> String {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default()
}

/// Resolve full path to a binary by checking common locations.
/// macOS .app bundles don't inherit the shell PATH, so we probe manually.
fn resolve_bin(name: &str) -> String {
    let home = home_dir();

    // Tauri installs the Windows ARM64 sidecar beside the desktop executable.
    // Prefer the bundled runtime over an unrelated Ollama install on PATH.
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    if name == "ollama" {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let sidecar = dir.join("ollama.exe");
                if sidecar.is_file() {
                    return sidecar.to_string_lossy().into_owned();
                }
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    let candidates = vec![
        format!("/opt/homebrew/bin/{name}"),
        format!("{home}/.local/bin/{name}"),
        format!("{home}/.cargo/bin/{name}"),
        format!("/usr/local/bin/{name}"),
        format!("/usr/bin/{name}"),
    ];

    #[cfg(target_os = "windows")]
    let candidates = {
        let localappdata = std::env::var("LOCALAPPDATA").unwrap_or_default();
        let programfiles = std::env::var("ProgramFiles").unwrap_or_default();
        let programfiles_x86 = std::env::var("ProgramFiles(x86)").unwrap_or_default();
        vec![
            // Git for Windows — standard install paths
            format!("{programfiles}\\Git\\cmd\\{name}.exe"),
            format!("{programfiles_x86}\\Git\\cmd\\{name}.exe"),
            format!("{localappdata}\\Programs\\Git\\cmd\\{name}.exe"),
            // Scoop package manager
            format!("{home}\\scoop\\shims\\{name}.exe"),
            // Cargo, local bin
            format!("{home}\\.cargo\\bin\\{name}.exe"),
            format!("{home}\\.local\\bin\\{name}.exe"),
            // Generic program locations
            format!("{localappdata}\\Programs\\{name}\\{name}.exe"),
            format!("{programfiles}\\{name}\\{name}.exe"),
            // Ollama installs to LOCALAPPDATA on Windows
            format!("{localappdata}\\Programs\\Ollama\\{name}.exe"),
            // uv installs via pip/pipx
            format!("{home}\\AppData\\Roaming\\Python\\Scripts\\{name}.exe"),
        ]
    };

    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return path.clone();
        }
    }

    // Fallback: ask the OS to find it on PATH.
    // On Windows this uses `where.exe`, on Unix `which`.
    #[cfg(target_os = "windows")]
    {
        if let Ok(output) = local_std_command("where")
            .arg(format!("{name}.exe"))
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(first_line) = stdout.lines().next() {
                    let p = first_line.trim();
                    if !p.is_empty() && std::path::Path::new(p).exists() {
                        return p.to_string();
                    }
                }
            }
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(output) = local_std_command("which").arg(name).output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(first_line) = stdout.lines().next() {
                    let p = first_line.trim();
                    if !p.is_empty() && std::path::Path::new(p).exists() {
                        return p.to_string();
                    }
                }
            }
        }
    }

    name.to_string()
}

/// Find the OpenJarvis project root (contains pyproject.toml).
/// Checks OPENJARVIS_ROOT env var, the checkout beside the executable,
/// then walks up from the executable before probing common clone locations.
fn find_project_root() -> Option<std::path::PathBuf> {
    // 1. Explicit env var override
    if let Ok(root) = std::env::var("OPENJARVIS_ROOT") {
        let path = std::path::PathBuf::from(&root);
        if path.join("pyproject.toml").exists() {
            return Some(path);
        }
    }

    // 2. Find the checkout associated with this executable before considering
    //    unrelated clones elsewhere on the machine.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(root) = project_root_near_executable(&exe) {
            return Some(root);
        }
    }

    // 3. Fallback: well-known direct paths
    let home = home_dir();
    let direct = [
        format!("{home}/OpenJarvis"),
        format!("{home}/projects/hazy/OpenJarvis"),
        format!("{home}/projects/OpenJarvis"),
        format!("{home}/src/OpenJarvis"),
        format!("{home}/Documents/OpenJarvis"),
        format!("{home}/Desktop/OpenJarvis"),
        format!("{home}/Developer/OpenJarvis"),
        format!("{home}/dev/OpenJarvis"),
        format!("{home}/Code/OpenJarvis"),
        format!("{home}/code/OpenJarvis"),
        format!("{home}/repos/OpenJarvis"),
        format!("{home}/github/OpenJarvis"),
    ];
    for p in &direct {
        let path = std::path::PathBuf::from(p);
        if path.join("pyproject.toml").exists() {
            return Some(path);
        }
    }

    // 4. Shallow scan: look for OpenJarvis one level inside common parent dirs.
    //    This catches clones like ~/Documents/my-stuff/OpenJarvis without
    //    needing to enumerate every possible intermediate folder.
    let scan_parents = [
        format!("{home}/Documents"),
        format!("{home}/Desktop"),
        format!("{home}/Developer"),
        format!("{home}/projects"),
        format!("{home}/repos"),
        format!("{home}/src"),
        format!("{home}/Code"),
        format!("{home}/code"),
        format!("{home}/dev"),
        format!("{home}/github"),
    ];
    for parent in &scan_parents {
        let parent_path = std::path::PathBuf::from(parent);
        if let Ok(entries) = std::fs::read_dir(&parent_path) {
            for entry in entries.flatten() {
                let candidate = entry.path().join("OpenJarvis");
                if candidate.join("pyproject.toml").exists() {
                    return Some(candidate);
                }
                // Also check if the entry itself is OpenJarvis (case-insensitive match)
                if let Some(name) = entry.file_name().to_str() {
                    if name.eq_ignore_ascii_case("openjarvis")
                        && entry.path().join("pyproject.toml").exists()
                    {
                        return Some(entry.path());
                    }
                }
            }
        }
    }

    None
}

fn project_root_near_executable(exe: &std::path::Path) -> Option<std::path::PathBuf> {
    let exe_dir = exe.parent()?;

    if exe_dir.join("pyproject.toml").is_file() {
        return Some(exe_dir.to_path_buf());
    }

    // The Windows installer places the desktop executable in OpenJarvis/ and
    // clones the matching Python backend into OpenJarvis/src/.
    let installed_checkout = exe_dir.join("src");
    if installed_checkout.join("pyproject.toml").is_file() {
        return Some(installed_checkout);
    }

    // Development builds and app bundles may sit inside the checkout.
    let mut dir = exe_dir.parent();
    for _ in 1..8 {
        let current = dir?;
        if current.join("pyproject.toml").is_file() {
            return Some(current.to_path_buf());
        }
        dir = current.parent();
    }

    None
}

// ---------------------------------------------------------------------------
// BackendManager — owns the Ollama + Jarvis server child processes
// ---------------------------------------------------------------------------

struct ChildHandle {
    child: tokio::process::Child,
}

impl ChildHandle {
    async fn kill(&mut self) {
        let _ = self.child.kill().await;
    }
}

/// Spawn a subprocess whose lifetime cannot escape the async operation that
/// created it.  Tokio's default is `kill_on_drop(false)`, which means aborting
/// the boot task in the small window between `spawn()` and storing the child in
/// `BackendManager` would detach it.  The same default also lets one-shot boot
/// commands (`git clone`, `uv sync`, extension verification) continue after a
/// source reset.  Every subprocess on the boot path must opt in here (or call
/// `kill_on_drop(true)` before `output()`).
fn spawn_owned_child(cmd: &mut tokio::process::Command) -> std::io::Result<tokio::process::Child> {
    cmd.kill_on_drop(true);
    cmd.spawn()
}

/// Rolling buffer holding the most recent ~16 KB of jarvis stderr.
///
/// Populated by a background drainer task spawned at boot so the pipe
/// never fills and back-pressures `jarvis serve`; consumed by the boot
/// path when surfacing failure messages.
type StderrTail = Arc<Mutex<Vec<u8>>>;

const STDERR_TAIL_LIMIT: usize = 16 * 1024;

struct BackendManager {
    ollama: Option<ChildHandle>,
    jarvis: Option<ChildHandle>,
    jarvis_stderr_tail: StderrTail,
    boot_task: Option<tauri::async_runtime::JoinHandle<()>>,
    bundled_root: Option<std::path::PathBuf>,
    attached: Option<offline::AttachedBackend>,
}

impl Default for BackendManager {
    fn default() -> Self {
        Self {
            ollama: None,
            jarvis: None,
            jarvis_stderr_tail: Arc::new(Mutex::new(Vec::new())),
            boot_task: None,
            bundled_root: None,
            attached: None,
        }
    }
}

impl BackendManager {
    async fn stop_all(&mut self) {
        if let Some(task) = self.boot_task.take() {
            task.abort();
            // `abort()` only requests cancellation.  Await it before touching
            // children or returning so the boot future has dropped any local,
            // not-yet-registered child and cannot write config after reset.
            let _ = task.await;
        }
        self.stop_children().await;
    }

    async fn stop_children(&mut self) {
        if let Some(ref mut h) = self.jarvis {
            h.kill().await;
        }
        self.jarvis = None;
        if let Some(ref mut h) = self.ollama {
            h.kill().await;
        }
        self.ollama = None;
    }
}

type SharedBackend = Arc<Mutex<BackendManager>>;

/// Spawn exactly one tracked boot task. Keeping its handle lets recovery
/// abort an in-flight endpoint check/model download before killing children.
async fn start_managed_boot(backend: SharedBackend, status: SharedStatus, try_inference: bool) {
    let mut manager = backend.lock().await;
    if let Some(task) = manager.boot_task.take() {
        task.abort();
        let _ = task.await;
    }
    let task_backend = backend.clone();
    let task = tauri::async_runtime::spawn(boot_backend(task_backend, status, try_inference));
    manager.boot_task = Some(task);
}

// ---------------------------------------------------------------------------
// Setup status (reported to frontend)
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, Clone)]
struct SetupStatus {
    phase: String,
    detail: String,
    ollama_ready: bool,
    server_ready: bool,
    model_ready: bool,
    error: Option<String>,
    /// The configured source or "unconfigured", for source-aware setup labels.
    source: String,
    /// A fresh desktop install must collect an explicit source choice before
    /// any inference process is allowed to start.
    requires_source: bool,
}

impl Default for SetupStatus {
    fn default() -> Self {
        Self {
            phase: "awaiting_source".into(),
            detail: "Choose where A.C.E. should run models.".into(),
            ollama_ready: false,
            server_ready: false,
            model_ready: false,
            error: None,
            source: "unconfigured".into(),
            requires_source: true,
        }
    }
}

impl SetupStatus {
    fn starting(cfg: &InferenceConfig) -> Self {
        Self {
            phase: "starting".into(),
            detail: "Initializing...".into(),
            source: cfg.kind.label().into(),
            requires_source: false,
            ..Self::default()
        }
    }
}

type SharedStatus = Arc<Mutex<SetupStatus>>;

// ---------------------------------------------------------------------------
// Health-check helpers
// ---------------------------------------------------------------------------

async fn wait_for_url(url: &str, timeout: Duration) -> bool {
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    let deadline = tokio::time::Instant::now() + timeout;
    while tokio::time::Instant::now() < deadline {
        if let Ok(resp) = client.get(url).send().await {
            if resp.status().is_success() {
                return true;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

/// True if a custom OpenAI-compatible endpoint answers at all (any HTTP
/// status counts — even a 404 proves the server is up). `host` is the bare
/// base URL; we probe `<host>/v1/models`.
async fn endpoint_reachable(host: &str, timeout: Duration, local: bool) -> bool {
    let builder = reqwest::Client::builder().timeout(Duration::from_secs(3));
    let builder = if local {
        builder.no_proxy().redirect(reqwest::redirect::Policy::none())
    } else {
        builder
    };
    let client = match builder.build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    let url = format!("{}/v1/models", host.trim_end_matches('/'));
    let deadline = tokio::time::Instant::now() + timeout;
    while tokio::time::Instant::now() < deadline {
        if client.get(&url).send().await.is_ok() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

/// Outcome of waiting for `jarvis serve` to become healthy.
///
/// Unlike [`wait_for_url`] this differentiates "server is up but degraded"
/// (HTTP 503 — usually inference engine failed to load) from "server never
/// came up" and from "child process died before serving anything", because
/// each needs a different user-facing message.
#[derive(Debug)]
enum JarvisStartResult {
    /// `/health` returned 2xx.
    Ready,
    /// Server replied 503. The body is the actionable message (typically
    /// "engine not ready" or a model-load error).
    ServiceUnavailable(String),
    /// The packaged server speaks a different API revision.
    IncompatiblePackagedApi,
    /// The `jarvis serve` child exited before `/health` returned 2xx.
    EarlyExit { code: Option<i32>, stderr: String },
    /// Deadline elapsed without ever seeing 2xx or an early exit.
    Timeout,
}

/// Spawn a detached task that continuously drains `jarvis serve`'s
/// stderr into a rolling tail buffer.
///
/// We MUST keep reading stderr for as long as the child runs — `jarvis
/// serve` is chatty (engine load progress, request logs), and the OS
/// pipe buffer is small (4 KB on Windows, 64 KB on Linux). Once full,
/// the child's next stderr write blocks indefinitely and the server
/// hangs mid-operation. The drainer reads in chunks and keeps only the
/// last `STDERR_TAIL_LIMIT` bytes — enough to surface a tail trace if
/// the child later dies, without unbounded memory growth.
///
/// Returns immediately after spawning the task; the task ends naturally
/// when the child closes stderr (i.e. exits).
fn spawn_jarvis_stderr_drainer(mut stderr: tokio::process::ChildStderr, tail: StderrTail) {
    use tokio::io::AsyncReadExt;
    tokio::spawn(async move {
        let mut buf = vec![0u8; 4096];
        loop {
            match stderr.read(&mut buf).await {
                Ok(0) => break,  // EOF — child closed stderr
                Err(_) => break, // pipe broke — also done
                Ok(n) => {
                    let mut t = tail.lock().await;
                    t.extend_from_slice(&buf[..n]);
                    if t.len() > STDERR_TAIL_LIMIT {
                        let drop_n = t.len() - STDERR_TAIL_LIMIT;
                        t.drain(..drop_n);
                    }
                }
            }
        }
    });
}

/// Read whatever the stderr drainer has buffered so far.
///
/// Safe to call at any time; returns an empty string before the
/// drainer has seen any bytes. Trimmed.
async fn read_jarvis_stderr_tail(backend: &SharedBackend) -> String {
    let tail = backend.lock().await.jarvis_stderr_tail.clone();
    let bytes = tail.lock().await.clone();
    String::from_utf8_lossy(&bytes).trim().to_string()
}

enum JarvisHealthTarget<'a> {
    Standard {
        url: &'a str,
        cloudflare: bool,
    },
    Packaged {
        runtime: &'a offline::BundledBackend,
        port: u16,
        mode: offline::AttachedBackendMode,
    },
}

/// Poll `jarvis serve` health, watching the child process state so we
/// never wait 10 minutes for a process that crashed in the first second.
async fn wait_for_jarvis_health(
    target: JarvisHealthTarget<'_>,
    timeout: Duration,
    backend: &SharedBackend,
) -> JarvisStartResult {
    let client = match reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(_) => return JarvisStartResult::Timeout,
    };
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        // 1. Has the child already exited? `try_wait` is non-blocking; on
        // Windows where uv / python / the Rust extension can fail to load
        // very fast, this catches the crash within ~500ms instead of after
        // the full HTTP timeout window.
        let exit_status = {
            let mut mgr = backend.lock().await;
            match mgr.jarvis.as_mut() {
                Some(h) => h.child.try_wait().ok().flatten(),
                None => None,
            }
        };
        if let Some(status) = exit_status {
            let stderr = read_jarvis_stderr_tail(backend).await;
            return JarvisStartResult::EarlyExit {
                code: status.code(),
                stderr,
            };
        }

        match &target {
            JarvisHealthTarget::Standard { url, cloudflare } => {
                if let Ok(resp) = client.get(*url).send().await {
                    if cloudflare_response(resp.headers()) != *cloudflare {
                        return JarvisStartResult::ServiceUnavailable(
                            "The inference source on the ACE port changed.".into(),
                        );
                    }
                    if resp.status().is_success() {
                        return JarvisStartResult::Ready;
                    }
                    if resp.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE {
                        let body = if *cloudflare {
                            "Cloudflare inference is unavailable.".into()
                        } else {
                            resp.text().await.unwrap_or_default()
                        };
                        return JarvisStartResult::ServiceUnavailable(body);
                    }
                }
            }
            JarvisHealthTarget::Packaged { runtime, port, mode } => {
                if let Some(result) = check_packaged_start_health(runtime, *port, *mode, backend).await {
                    return result;
                }
            }
        }

        if tokio::time::Instant::now() >= deadline {
            return JarvisStartResult::Timeout;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

fn parse_ollama_models(body: &serde_json::Value) -> Vec<serde_json::Value> {
    body.get("models")
        .and_then(serde_json::Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|model| {
                    let name = model
                        .get("name")
                        .or_else(|| model.get("model"))
                        .and_then(serde_json::Value::as_str)?;
                    if name.trim().is_empty() {
                        return None;
                    }
                    let size = model
                        .get("size")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or_default();
                    Some(serde_json::json!({"name": name, "size": size}))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_ollama_model_names(body: &serde_json::Value) -> Vec<String> {
    parse_ollama_models(body)
        .iter()
        .filter_map(|model| model.get("name").and_then(serde_json::Value::as_str))
        .map(str::to_string)
        .collect()
}

fn model_names_match(installed: &str, requested: &str) -> bool {
    installed == requested
        || installed.strip_suffix(":latest") == Some(requested)
        || requested.strip_suffix(":latest") == Some(installed)
}

fn matching_installed_model(models: &[String], requested: &str) -> Option<String> {
    models
        .iter()
        .find(|model| model_names_match(model, requested))
        .cloned()
}

fn normalize_ollama_model_tag(model: &str) -> Result<String, String> {
    let model = model.trim();
    let bytes = model.as_bytes();
    let valid = !bytes.is_empty()
        && bytes.len() <= 128
        && bytes[0].is_ascii_alphanumeric()
        && bytes[1..].iter().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(*byte, b'.' | b'_' | b':' | b'/' | b'-')
        });
    if valid {
        Ok(model.to_string())
    } else {
        Err(
            "Enter a valid Ollama model tag using letters, numbers, '.', '_', ':', '/', or '-'."
                .into(),
        )
    }
}

async fn ollama_model_names() -> Vec<String> {
    let url = format!("http://127.0.0.1:{}/api/tags", OLLAMA_PORT);
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
    {
        Ok(client) => client,
        Err(_) => return Vec::new(),
    };
    if let Ok(resp) = client.get(&url).send().await {
        if let Ok(body) = resp.json::<serde_json::Value>().await {
            return parse_ollama_model_names(&body);
        }
    }
    Vec::new()
}

async fn fetch_ollama_models() -> Result<Vec<serde_json::Value>, String> {
    let url = format!("http://127.0.0.1:{}/api/tags", OLLAMA_PORT);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .no_proxy()
        .build()
        .map_err(|error| format!("Could not create local Ollama client: {error}"))?;
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|error| format!("Ollama is unavailable at {url}: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("Ollama returned HTTP {}", response.status()));
    }
    let body = response
        .json::<serde_json::Value>()
        .await
        .map_err(|error| format!("Ollama returned invalid model data: {error}"))?;
    Ok(parse_ollama_models(&body))
}

async fn pull_model(model: &str) -> Result<(), String> {
    let model = normalize_ollama_model_tag(model)?;
    let url = format!("http://127.0.0.1:{}/api/pull", OLLAMA_PORT);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .post(&url)
        .json(&serde_json::json!({"name": model, "stream": false}))
        .send()
        .await
        .map_err(|e| format!("Pull request failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("Pull returned status {}", resp.status()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// uv sync error formatting (pure helpers — unit-tested, see #331)
// ---------------------------------------------------------------------------

/// Last `max_chars` characters of a `uv sync` stderr stream, trimmed.
///
/// uv's actionable diagnostic almost always lands at the tail of the
/// stream, so when surfacing a failure to the user we show the end, not
/// the (usually noisy progress-spinner) beginning. Operates on `char`
/// boundaries so it never splits a multi-byte UTF-8 codepoint — important
/// because Windows consoles emit non-ASCII (cp9xx) bytes.
fn uv_sync_stderr_tail(stderr: &str, max_chars: usize) -> String {
    let total = stderr.chars().count();
    let skip = total.saturating_sub(max_chars);
    stderr
        .chars()
        .skip(skip)
        .collect::<String>()
        .trim()
        .to_string()
}

/// Error message shown when `uv sync` runs but exits non-zero (#331).
///
/// `exit_code` is `None` when the process was terminated by a signal with
/// no exit code (rendered as "unknown" rather than a misleading -1).
fn format_uv_sync_failure(root: &std::path::Path, exit_code: Option<i32>, stderr: &str) -> String {
    let code = exit_code
        .map(|c| c.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let tail = uv_sync_stderr_tail(stderr, 800);
    let rust_hint = if looks_like_rust_extension_build_error(stderr) {
        format!("\n\n{}", rust_toolchain_install_hint())
    } else {
        String::new()
    };
    format!(
        "`uv sync` failed in {} (exit {}). Last output:\n\n{}\n\n\
         Try opening a terminal in that directory and running \
         `{}` manually for the full output.{}",
        root.display(),
        code,
        tail,
        DESKTOP_UV_SYNC_COMMAND,
        rust_hint,
    )
}

/// Strip AppImage-injected environment from a subprocess command (#455).
///
/// When the OpenJarvis desktop binary is shipped as an AppImage, the AppImage
/// runtime sets `LD_LIBRARY_PATH` (and friends) to the extracted-to-/tmp
/// bundled lib dir. Any child we spawn inherits that env by default — but the
/// children we spawn (`uv`, `ollama`, `git`) live outside the AppImage and
/// must NOT load their shared libraries from the AppImage's bundle. The
/// classic symptom: `uv` finds `python3`, `python3` tries to `import numpy`,
/// numpy's `.so` files try to dlopen libstdc++/libssl/libcrypto, the linker
/// picks the AppImage's versions which were built against a different glibc
/// or libcrypto API, and python dies silently — before any startup log
/// reaches us. The user sees "API Server — starting server..." forever.
///
/// Fix: when we detect we're inside an AppImage (the AppImage runtime sets
/// `$APPIMAGE` to the original image path), strip the leaked env vars before
/// spawn. Conditional on `APPIMAGE` being set so regular Linux installs that
/// legitimately use `LD_LIBRARY_PATH` are untouched. Linux-only — the
/// `#[cfg]` makes this a no-op on macOS / Windows.
#[cfg_attr(not(target_os = "linux"), allow(unused_variables))]
fn prepare_subprocess_for_appimage(cmd: &mut tokio::process::Command) {
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("APPIMAGE").is_some() {
            cmd.env_remove("LD_LIBRARY_PATH");
            cmd.env_remove("LD_PRELOAD");
            cmd.env_remove("APPIMAGE");
            cmd.env_remove("APPIMAGE_UUID");
            cmd.env_remove("APPDIR");
            cmd.env_remove("ARGV0");
        }
    }
}

/// Error message shown when `uv sync` can't even be spawned (#331) —
/// e.g. the resolved `uv` binary doesn't exist or isn't executable.
fn format_uv_sync_spawn_error(root: &std::path::Path, uv_bin: &str, err: &str) -> String {
    format!(
        "Could not run `uv sync`: {}. Verify uv is installed at \
         `{}` and the OpenJarvis repo is at `{}`.",
        err,
        uv_bin,
        root.display(),
    )
}

fn rust_toolchain_install_hint() -> &'static str {
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    return "The desktop app needs Rust and Visual Studio Build Tools with the x64 C++ workload \
            for its x64 Python backend. Install Rust from https://rustup.rs, run \
            `rustup target add --toolchain stable x86_64-pc-windows-msvc`, then relaunch.";
    #[cfg(not(all(target_os = "windows", target_arch = "aarch64")))]
    "The desktop app needs the Rust toolchain to build `openjarvis_rust`. \
     Install Rust from https://rustup.rs. On Windows, also install Visual Studio \
     Build Tools with the C++ workload, then relaunch."
}

/// Windows ARM64 runs the current Python dependency stack under x64 emulation:
/// the locked numpy/pyarrow and speech wheels do not support win_arm64.
/// The Tauri shell and bundled Ollama remain native ARM64 processes.
#[cfg_attr(
    not(all(target_os = "windows", target_arch = "aarch64")),
    allow(unused_variables)
)]
fn prepare_backend_python(cmd: &mut tokio::process::Command) {
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        cmd.env("UV_PYTHON", "cpython-3.12-windows-x86_64-none");
        // The PyO3 module must match the interpreter, not the desktop shell.
        cmd.env("CARGO_BUILD_TARGET", "x86_64-pc-windows-msvc");
        // Use the same stable compiler where the x64 target was installed,
        // rather than selecting a separate toolchain through the MSRV pin.
        cmd.env("RUSTUP_TOOLCHAIN", "stable");
        // PyO3 cannot infer the target interpreter's ABI across host arches.
        cmd.env("PYO3_CROSS_PYTHON_VERSION", "3.12");
        cmd.env("PYO3_CROSS_PYTHON_IMPLEMENTATION", "CPython");
    }
}

fn looks_like_rust_extension_build_error(stderr: &str) -> bool {
    let lower = stderr.to_ascii_lowercase();
    [
        "openjarvis-rust",
        "openjarvis_rust",
        "maturin",
        "cargo",
        "rustc",
        "link.exe",
        "visual studio",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

fn format_missing_rust_toolchain() -> String {
    format!(
        "Could not find Rust's `cargo` command. {}\n\n\
         If Rust is already installed, close and relaunch the desktop app so \
         PATH includes `~/.cargo/bin`.",
        rust_toolchain_install_hint(),
    )
}

fn format_extension_import_failure(root: &std::path::Path, stderr: &str) -> String {
    let tail = uv_sync_stderr_tail(stderr, 4000);
    format!(
        "`openjarvis_rust` is still not importable after building. Last output:\n\n{}\n\n\
         Run these manually for the full build log:\n\n\
           cd {}\n\
           {}\n\
           uv run python -c \"import openjarvis_rust\"",
        if tail.is_empty() {
            "(no stderr output)"
        } else {
            &tail
        },
        root.display(),
        DESKTOP_UV_SYNC_COMMAND,
    )
}

fn add_cargo_bin_to_path(cmd: &mut tokio::process::Command) {
    let mut paths: Vec<std::path::PathBuf> = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect())
        .unwrap_or_default();
    paths.insert(
        0,
        std::path::PathBuf::from(home_dir())
            .join(".cargo")
            .join("bin"),
    );
    if let Ok(joined) = std::env::join_paths(paths) {
        cmd.env("PATH", joined);
    }
}

async fn verify_openjarvis_rust_extension(
    root: &std::path::Path,
    uv_bin: &str,
) -> Result<(), String> {
    let mut cmd = local_command(uv_bin);
    cmd.args(["run", "python", "-c", "import openjarvis_rust"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .current_dir(root);
    prepare_subprocess_for_appimage(&mut cmd);
    prepare_backend_python(&mut cmd);
    add_cargo_bin_to_path(&mut cmd);
    cmd.kill_on_drop(true);

    match cmd.output().await {
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            Err(format_extension_import_failure(root, &stderr))
        }
        Err(e) => Err(format!(
            "Could not verify `openjarvis_rust`: {}. Verify uv is installed at `{}`.",
            e, uv_bin
        )),
    }
}

fn port_owner_hint() -> String {
    if cfg!(target_os = "windows") {
        format!("netstat -ano | findstr :{}", JARVIS_PORT)
    } else {
        format!("lsof -i :{}", JARVIS_PORT)
    }
}

fn format_port_unavailable(port: u16, reason: &str) -> String {
    format!(
        "Port {} is not available: {}. Stop the process using that port or \
         change the OpenJarvis port, then relaunch.\n\nTo identify it:\n  {}",
        port,
        reason,
        port_owner_hint(),
    )
}

fn check_jarvis_port_available() -> Result<(), String> {
    match std::net::TcpListener::bind(("127.0.0.1", JARVIS_PORT)) {
        Ok(listener) => {
            drop(listener);
            Ok(())
        }
        Err(err) => Err(format_port_unavailable(JARVIS_PORT, &err.to_string())),
    }
}

// ---------------------------------------------------------------------------
// Backend boot sequence (runs in background after app launch)
// ---------------------------------------------------------------------------

async fn boot_backend(backend: SharedBackend, status: SharedStatus, try_inference: bool) {
    if !is_offline_distribution() {
        boot_configured_backend(backend, status, None).await;
        return;
    }

    *status.lock().await = SetupStatus {
        phase: "server".into(),
        detail: "Starting the packaged ACE backend ...".into(),
        requires_source: false,
        ..SetupStatus::default()
    };
    let root = backend.lock().await.bundled_root.clone();
    let runtime = match root.as_deref().map(offline::BundledBackend::open) {
        Some(Ok(runtime)) => runtime,
        Some(Err(error)) => {
            status.lock().await.error = Some(error);
            return;
        }
        None => {
            status.lock().await.error = Some("The packaged ACE backend could not be located.".into());
            return;
        }
    };
    if let Err(port_error) = check_jarvis_port_available() {
        if !try_attach_bundled_server(&runtime, JARVIS_PORT, &backend, &status, try_inference).await {
            status.lock().await.error = Some(port_error);
        }
        return;
    }

    backend.lock().await.attached = None;
    if let Err(error) = runtime.initialize_profile().await {
        status.lock().await.error = Some(error);
        return;
    }

    let inference_error = if try_inference && read_configured_inference_config().is_some() {
        boot_configured_backend(backend.clone(), status.clone(), Some(&runtime)).await;
        let state = status.lock().await.clone();
        if state.phase == "ready" {
            return;
        }
        backend.lock().await.stop_children().await;
        state.error
    } else {
        None
    };
    let source = read_configured_inference_config()
        .map_or("unconfigured", |cfg| cfg.kind.label());
    *status.lock().await = SetupStatus {
        phase: "server".into(),
        detail: "Starting local memory and knowledge services ...".into(),
        source: source.into(),
        requires_source: false,
        ..SetupStatus::default()
    };
    match start_bundled_server(&runtime, &backend, BundledSource::LocalServices).await {
        Ok(()) => {
            let mut state = status.lock().await;
            state.phase = "local_services_ready".into();
            state.server_ready = true;
            state.detail = inference_error.unwrap_or_else(|| {
                "Local services are ready. Choose an installed model to use chat.".into()
            });
        }
        Err(error) => {
            backend.lock().await.stop_children().await;
            status.lock().await.error = Some(error);
        }
    }
}

async fn spawn_managed_server(
    backend: &SharedBackend,
    cmd: &mut tokio::process::Command,
    capture_stderr: bool,
) -> std::io::Result<()> {
    let mut child = spawn_owned_child(
        cmd.stdout(std::process::Stdio::null())
            .stderr(if capture_stderr {
                std::process::Stdio::piped()
            } else {
                std::process::Stdio::null()
            }),
    )?;
    let stderr = child.stderr.take();
    let tail = Arc::new(Mutex::new(Vec::new()));
    {
        let mut manager = backend.lock().await;
        manager.jarvis = Some(ChildHandle { child });
        manager.jarvis_stderr_tail = tail.clone();
    }
    if let Some(stderr) = stderr {
        spawn_jarvis_stderr_drainer(stderr, tail);
    }
    Ok(())
}

enum BundledSource<'a> {
    LocalServices,
    LocalInference(&'a [String]),
    Cloudflare {
        args: &'a [String],
        credentials: &'a CloudflareEnvironment,
    },
}

fn configure_bundled_server(
    cmd: &mut tokio::process::Command,
    port: u16,
    source: BundledSource<'_>,
) -> offline::AttachedBackendMode {
    use offline::AttachedBackendMode;
    cmd.args(["serve", "--host", "127.0.0.1", "--port"])
        .arg(port.to_string());
    match source {
        BundledSource::LocalServices => {
            cmd.arg("--local-services-only");
            AttachedBackendMode::LocalServices
        }
        BundledSource::LocalInference(args) => {
            cmd.args(args).arg("--local-inference-only");
            AttachedBackendMode::Inference
        }
        BundledSource::Cloudflare { args, credentials } => {
            cmd.args(args);
            configure_cloudflare_environment(cmd, credentials);
            AttachedBackendMode::Cloudflare
        }
    }
}

async fn start_bundled_server(
    runtime: &offline::BundledBackend,
    backend: &SharedBackend,
    source: BundledSource<'_>,
) -> Result<(), String> {
    use offline::AttachedBackendMode;
    check_jarvis_port_available()?;
    let mut cmd = runtime.command();
    let mode = configure_bundled_server(&mut cmd, JARVIS_PORT, source);
    let cloudflare = mode == AttachedBackendMode::Cloudflare;
    spawn_managed_server(backend, &mut cmd, !cloudflare)
        .await
        .map_err(|err| format!("Could not launch the packaged ACE backend: {err}"))?;

    let local_services_only = mode == AttachedBackendMode::LocalServices;
    let timeout = if local_services_only { 30 } else { 120 };
    let target = JarvisHealthTarget::Packaged {
        runtime,
        port: JARVIS_PORT,
        mode,
    };
    match wait_for_jarvis_health(target, Duration::from_secs(timeout), backend).await {
        JarvisStartResult::Ready => Ok(()),
        JarvisStartResult::IncompatiblePackagedApi => Err(
            "The installed backend API does not match this version. Reinstall A.C.E."
                .into(),
        ),
        JarvisStartResult::ServiceUnavailable(_) if local_services_only => {
            Err("The local memory and knowledge services are unavailable.".into())
        }
        JarvisStartResult::ServiceUnavailable(_) if cloudflare => {
            Err("Cloudflare inference is unavailable. Check your Cloudflare credentials and connection.".into())
        }
        JarvisStartResult::ServiceUnavailable(_) => {
            Err("The selected local inference model is unavailable.".into())
        }
        JarvisStartResult::EarlyExit { .. } if cloudflare => {
            Err("Cloudflare could not start. Check your Cloudflare credentials and connection.".into())
        }
        JarvisStartResult::EarlyExit { code, .. } => {
            Err(format!("The packaged ACE backend exited before becoming ready (code {code:?})."))
        }
        JarvisStartResult::Timeout => Err(format!(
            "The packaged ACE backend did not become ready within {timeout} seconds."
        )),
    }
}

async fn boot_configured_backend(
    backend: SharedBackend,
    status: SharedStatus,
    bundled: Option<&offline::BundledBackend>,
) {
    // Never infer consent from a missing or malformed config. A fresh desktop
    // install must persist an explicit source choice before this function can
    // launch Ollama, contact a custom endpoint, or download a model.
    let Some(mut cfg) = read_configured_inference_config() else {
        *status.lock().await = SetupStatus::default();
        return;
    };
    let mut pending_rollback = PendingInferenceRollback::new(&cfg);
    let use_cloudflare = cfg.kind == SourceKind::Cloudflare;
    if bundled.is_some() && !offline_source_allowed(&cfg) {
        let mut state = status.lock().await;
        *state = SetupStatus::starting(&cfg);
        state.error = Some("Choose a supported local inference source or the Cloudflare Workers AI option.".into());
        return;
    }
    let plan = match boot_plan(&cfg) {
        Ok(plan) => plan,
        Err(error) => {
            let mut s = status.lock().await;
            *s = SetupStatus::starting(&cfg);
            s.error = Some(error);
            return;
        }
    };
    {
        let mut s = status.lock().await;
        *s = SetupStatus::starting(&cfg);
    }

    let cloudflare_credentials = if use_cloudflare {
        match cloudflare_credentials::environment().await {
            Ok(credentials) => Some(credentials),
            Err(error) => {
                status.lock().await.error = Some(error);
                return;
            }
        }
    } else {
        None
    };

    // For Ollama, an installed `:latest` alias may resolve to the same exact
    // user-selected tag. None on the custom path.
    let mut serve_model_override: Option<String> = None;

    if plan.launch_ollama {
        // Phase 1: Start Ollama
        {
            let mut s = status.lock().await;
            s.phase = "ollama".into();
            s.detail = "Starting inference engine...".into();
        }

        // Try the bundled sidecar first, fall back to system ollama
        let ollama_child = {
            let ollama_bin = resolve_bin("ollama");
            let mut sidecar_cmd = local_command(&ollama_bin);
            sidecar_cmd
                .arg("serve")
                .env("OLLAMA_HOST", format!("127.0.0.1:{}", OLLAMA_PORT))
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null());
            // Avoid LD_LIBRARY_PATH leak when running inside an AppImage (#455).
            prepare_subprocess_for_appimage(&mut sidecar_cmd);
            match spawn_owned_child(&mut sidecar_cmd) {
                Ok(child) => Some(child),
                Err(_) => None,
            }
        };

        if let Some(child) = ollama_child {
            backend.lock().await.ollama = Some(ChildHandle { child });
        }

        let ollama_url = format!("http://127.0.0.1:{}/api/tags", OLLAMA_PORT);
        if !wait_for_url(&ollama_url, Duration::from_secs(30)).await {
            let mut s = status.lock().await;
            s.error = Some("Could not start Ollama. Install it from https://ollama.com".into());
            return;
        }

        {
            let mut s = status.lock().await;
            s.ollama_ready = true;
            s.detail = "Inference engine ready.".into();
        }

        // Phase 2: use only the model tag explicitly selected in Settings.
        // Missing tags are rejected by `boot_plan` before Ollama starts.
        let model = plan
            .model_to_pull
            .clone()
            .expect("an Ollama boot plan always contains its validated selected model");
        {
            let mut s = status.lock().await;
            s.phase = "model".into();
            s.detail = format!("Checking for {}...", model);
        }

        let installed_models = ollama_model_names().await;
        let resolved_model = if let Some(installed) =
            matching_installed_model(&installed_models, &model)
        {
            installed
        } else {
            if bundled.is_some() {
                status.lock().await.error = Some(format!(
                    "Ollama model '{model}' is not installed. Choose an installed or imported model in Settings."
                ));
                return;
            }
            {
                let mut s = status.lock().await;
                s.detail = format!("Downloading {}... (this may take a minute)", model);
            }
            match pull_model(&model).await {
                Ok(()) => model.clone(),
                Err(e) => {
                    let mut s = status.lock().await;
                    s.error = Some(format!(
                        "Could not install the selected Ollama model '{}': {}. \
                         Check the tag or network connection, then choose another model in Settings.",
                        model, e
                    ));
                    return;
                }
            }
        };

        if resolved_model != model {
            let mut s = status.lock().await;
            s.detail = format!("Using installed model {}.", resolved_model);
        }

        serve_model_override = Some(resolved_model.clone());

        {
            let mut s = status.lock().await;
            s.model_ready = true;
            s.detail = "Model ready.".into();
        }
    } else if !use_cloudflare {
        // Custom OpenAI-compatible endpoint: never start Ollama, never download.
        let host = plan
            .engine_host
            .as_ref()
            .map(|(_, v)| v.clone())
            .unwrap_or_default();
        {
            let mut s = status.lock().await;
            s.phase = "model".into();
            s.detail = format!("Connecting to {}...", host);
        }
        if host.is_empty() || !endpoint_reachable(&host, Duration::from_secs(15), bundled.is_some()).await {
            let mut s = status.lock().await;
            s.error = Some(format!(
                "Could not reach your custom inference server at {}. \
                 Start the server (e.g. LM Studio), or choose Change inference source below.",
                if host.is_empty() {
                    "(no URL set)"
                } else {
                    host.as_str()
                }
            ));
            return;
        }
        // Point `jarvis serve` at the user's endpoint by writing the engine
        // host into ~/.openjarvis/config.toml (the env var alone is shadowed by
        // the engine's non-empty default host in the Python layer).
        if bundled.is_none() {
            if let Some((engine, host)) = &plan.engine_host {
                if let Err(e) = set_engine_host_in_config(engine, host) {
                    let mut s = status.lock().await;
                    s.error = Some(format!("Could not write engine config: {}", e));
                    return;
                }
            }
        }
        {
            let mut s = status.lock().await;
            s.ollama_ready = true;
            s.model_ready = true;
            s.detail = "Connected to custom endpoint.".into();
        }
    }

    // Phase 3: Start jarvis serve
    {
        let mut s = status.lock().await;
        s.phase = "server".into();
        s.detail = "Starting API server...".into();
    }

    if let Some(runtime) = bundled {
        let args = offline_serve_args(&plan, serve_model_override.as_deref());
        let source = match &cloudflare_credentials {
            Some(credentials) => BundledSource::Cloudflare { args: &args, credentials },
            None => BundledSource::LocalInference(&args),
        };
        if let Err(error) = start_bundled_server(runtime, &backend, source).await {
            status.lock().await.error = Some(error);
            return;
        }
        if let Err(error) = pending_rollback.confirm(&mut cfg) {
            status.lock().await.error = Some(format!("Could not confirm inference setup: {error}"));
            return;
        }
        let mut state = status.lock().await;
        state.server_ready = true;
        state.model_ready = true;
        state.ollama_ready = true;
        state.phase = "ready".into();
        state.detail = if use_cloudflare {
            "Cloudflare inference ready."
        } else {
            "Local model and API ready."
        }.into();
        return;
    }

    let uv_bin = resolve_bin("uv");

    // Verify uv is actually installed. Concrete per-OS instructions —
    // the generic "install it from astral.sh" was the #1 source of
    // confusion on the Discord support thread; users couldn't tell whether
    // to use winget, scoop, pip, or the official installer.
    if !std::path::Path::new(&uv_bin).exists() && uv_bin == "uv" {
        let mut s = status.lock().await;
        #[cfg(target_os = "windows")]
        let msg = "Could not find 'uv' (Python package manager). \
                   To install on Windows, open PowerShell and run:\n\n\
                   powershell -ExecutionPolicy Bypass -c \"irm https://astral.sh/uv/install.ps1 | iex\"\n\n\
                   Then close and relaunch this app. \
                   (If the install completes but the app still can't find uv, \
                   you may need to log out and back in so PATH refreshes.)";
        #[cfg(target_os = "macos")]
        let msg = "Could not find 'uv' (Python package manager). \
                   To install on macOS, open Terminal and run:\n\n\
                   curl -LsSf https://astral.sh/uv/install.sh | sh\n\n\
                   Then relaunch this app.";
        #[cfg(target_os = "linux")]
        let msg = "Could not find 'uv' (Python package manager). \
                   To install on Linux, open a terminal and run:\n\n\
                   curl -LsSf https://astral.sh/uv/install.sh | sh\n\n\
                   Then relaunch this app.";
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        let msg = "Could not find 'uv' (Python package manager). \
                   Install it from https://astral.sh/uv then relaunch.";
        s.error = Some(msg.into());
        return;
    }

    let mut project_root = find_project_root();

    if project_root.is_none() {
        // Auto-clone on first launch
        let git_bin = resolve_bin("git");

        // Check that git is installed
        if !std::path::Path::new(&git_bin).exists() && git_bin == "git" {
            let mut s = status.lock().await;
            s.error = Some(
                "Could not find 'git'. \
                 Install it from https://git-scm.com then relaunch."
                    .into(),
            );
            return;
        }

        let target_path = std::path::PathBuf::from(home_dir()).join("OpenJarvis");
        let clone_target = target_path.display().to_string();

        // If the directory exists but is not a valid project, don't overwrite
        if target_path.exists() && !target_path.join("pyproject.toml").exists() {
            let mut s = status.lock().await;
            s.error = Some(format!(
                "{} exists but is not a valid OpenJarvis project. \
                 Remove it and relaunch, or set OPENJARVIS_ROOT to the correct path.",
                clone_target,
            ));
            return;
        }

        {
            let mut s = status.lock().await;
            s.detail = "Downloading OpenJarvis (first launch)...".into();
        }

        let mut clone_cmd = local_command(&git_bin);
        clone_cmd
            .args([
                "clone",
                "--depth",
                "1",
                "https://github.com/open-jarvis/OpenJarvis.git",
                &clone_target,
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped());
        prepare_subprocess_for_appimage(&mut clone_cmd);
        let clone_result = spawn_owned_child(&mut clone_cmd);

        match clone_result {
            Ok(child) => match child.wait_with_output().await {
                Ok(output) if output.status.success() => {
                    project_root = Some(target_path);
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let mut s = status.lock().await;
                    s.error = Some(format!(
                        "Failed to download OpenJarvis: {}. \
                         Clone manually: git clone https://github.com/open-jarvis/OpenJarvis.git {}",
                        stderr.trim(),
                        clone_target,
                    ));
                    return;
                }
                Err(e) => {
                    let mut s = status.lock().await;
                    s.error = Some(format!(
                        "Failed to download OpenJarvis: {}. \
                         Clone manually: git clone https://github.com/open-jarvis/OpenJarvis.git {}",
                        e, clone_target,
                    ));
                    return;
                }
            },
            Err(e) => {
                let mut s = status.lock().await;
                s.error = Some(format!(
                    "Could not run git: {}. \
                     Install git from https://git-scm.com then relaunch.",
                    e,
                ));
                return;
            }
        }
    }

    // Cloudflare must start an owned loopback worker with this launch's credentials.
    if use_cloudflare {
        if let Err(error) = check_jarvis_port_available() {
            status.lock().await.error = Some(error);
            return;
        }
    }

    // If something is already serving on our port, decide what to do based
    // on what it actually responds with — don't blindly kill it (#455).
    //
    // The OLD behaviour was: any HTTP response (even 404) → `fuser -k 8000/tcp`
    // / `taskkill /PID /F`. That broke the legitimate case where a user had
    // already started `jarvis serve` in a terminal and then launched the
    // desktop app — the app killed their server, then raced to spawn its
    // own, sometimes losing the race and hanging.
    //
    // New behaviour, by response shape:
    //   * 2xx /health        — healthy jarvis serve. Attach to it; skip the
    //                          uv-sync + spawn dance entirely. Done.
    //   * 503                — server is up but engine isn't ready. Surface
    //                          an actionable message; don't kill (matches
    //                          our wait_for_jarvis_health 503 contract).
    //   * any other status   — something else is listening on the port. Tell
    //                          the user via the error banner instead of
    //                          force-killing a foreign service.
    //   * Err (conn refused) — nothing is listening. Proceed to spawn.
    //
    // TODO(#455 follow-up): validate /health response body before attaching
    // so a multi-user host can't trivially spoof us. Also accept a port
    // override from config instead of hard-coding JARVIS_PORT.
    if !use_cloudflare {
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        match client
            .get(format!("http://127.0.0.1:{}/health", JARVIS_PORT))
            .send()
            .await
        {
            Ok(resp) if cloudflare_response(resp.headers()) => {
                status.lock().await.error = Some(
                    "Cloudflare is running on the ACE port. Stop that backend before starting a local or custom source.".into(),
                );
                return;
            }
            Ok(resp) if resp.status().is_success() => {
                // Confirm with a second probe — the first might have caught
                // a flickering server (engine half-loaded, dying mid-stop,
                // etc.) and we don't want to claim ready off a 2-second
                // snapshot. Small sleep between to give the server room.
                tokio::time::sleep(Duration::from_millis(500)).await;
                let confirm = client
                    .get(format!("http://127.0.0.1:{}/health", JARVIS_PORT))
                    .send()
                    .await
                    .map(|r| r.status().is_success() && !cloudflare_response(r.headers()))
                    .unwrap_or(false);
                if !confirm {
                    // First probe was 2xx but the second wasn't — fall
                    // through to the spawn path. The server probably went
                    // away between probes.
                    // (No early return — we want to spawn our own.)
                } else {
                    // A newly staged key may only enter a child process that
                    // this desktop instance launched. A listener that merely
                    // answers /health could be a port squatter; never POST or
                    // otherwise disclose the pending secret to it.
                    match pending_rollback.staged_credential_present() {
                        Ok(true) => {
                            let mut s = status.lock().await;
                            s.error = Some(format!(
                                "An API server is already running on port {}, but OpenJarvis cannot securely apply your new inference API key to a server it did not start. Stop that server, then try setup again.",
                                JARVIS_PORT,
                            ));
                            return;
                        }
                        Err(err) => {
                            let mut s = status.lock().await;
                            s.error = Some(format!(
                                "Could not verify the staged inference API key: {}",
                                err
                            ));
                            return;
                        }
                        Ok(false) => {}
                    }
                    // Attach to the existing healthy server. Mark every
                    // pre-spawn step done so the setup UI doesn't show a
                    // half-progress bar (model_ready / ollama_ready stay
                    // false otherwise because we skipped those steps).
                    if let Err(err) = pending_rollback.confirm(&mut cfg) {
                        let mut s = status.lock().await;
                        s.error = Some(format!("Could not confirm inference setup: {}", err));
                        return;
                    }
                    let mut s = status.lock().await;
                    s.phase = "ready".into();
                    s.detail =
                        format!("Connected to existing API server on port {}.", JARVIS_PORT,);
                    s.server_ready = true;
                    s.model_ready = true;
                    s.ollama_ready = true;
                    return;
                }
            }
            Ok(resp) if resp.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE => {
                let mut s = status.lock().await;
                s.error = Some(format!(
                    "An API server is already running on port {} but its \
                     inference engine isn't ready (HTTP 503). If this is your \
                     `jarvis serve`, wait for it to finish loading and relaunch. \
                     Otherwise, stop that service or change the port.",
                    JARVIS_PORT,
                ));
                return;
            }
            Ok(resp) => {
                // Something else (a different web server, a stale process,
                // a 4xx-returning instance) is on our port. Don't kill it —
                // give the user actionable info instead.
                let mut s = status.lock().await;
                s.error = Some(format!(
                    "Port {} is already in use by another service (it answered \
                     /health with HTTP {}). Stop that service or change the \
                     OpenJarvis port, then relaunch.\n\nTo identify it:\n  {}",
                    JARVIS_PORT,
                    resp.status(),
                    port_owner_hint(),
                ));
                return;
            }
            Err(_) => {
                // Nothing listening — proceed to the normal spawn path.
            }
        }
    }

    if let Err(err) = check_jarvis_port_available() {
        let mut s = status.lock().await;
        s.error = Some(err);
        return;
    }

    let root = project_root.as_ref().unwrap();

    let cargo_bin = resolve_bin("cargo");
    if !std::path::Path::new(&cargo_bin).exists() && cargo_bin == "cargo" {
        let mut s = status.lock().await;
        s.error = Some(format_missing_rust_toolchain());
        return;
    }

    // Install dependencies automatically (handles fresh clones).
    //
    // Previously we ran `uv sync` with both stdout AND stderr piped to
    // /dev/null and discarded the exit code (`let _ = …`). When `uv sync`
    // failed — Windows path issues, network problems, lockfile conflicts —
    // the user saw no error, the boot continued, `uv run jarvis serve`
    // then ran in an under-provisioned venv, and the user waited the full
    // 600s health-check window before getting "Jarvis server did not
    // become healthy in time" with no actionable detail (issue #331).
    //
    // Now: capture stderr, check the exit status, surface a useful error
    // to the user BEFORE the long server-start wait. The status detail
    // message also indicates this can take a couple of minutes on first
    // boot so users don't restart the app thinking it's stuck.
    {
        let mut s = status.lock().await;
        s.detail = "Installing dependencies (uv sync — may take 1-2 min on first boot)...".into();
    }
    let mut sync_cmd = local_command(&uv_bin);
    sync_cmd
        .args([
            "sync",
            "--extra",
            "desktop",
            "--extra",
            "inference-cloud",
            "--extra",
            "inference-google",
            // openjarvis_rust lives in a uv dependency group (not the published
            // `desktop` extra) so pip installs from PyPI don't require it (#584).
            "--group",
            "desktop-native",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .current_dir(root);
    // Avoid LD_LIBRARY_PATH leak when running inside an AppImage (#455).
    prepare_subprocess_for_appimage(&mut sync_cmd);
    prepare_backend_python(&mut sync_cmd);
    add_cargo_bin_to_path(&mut sync_cmd);
    sync_cmd.kill_on_drop(true);
    let sync_output = sync_cmd.output().await;
    match sync_output {
        Ok(out) if !out.status.success() => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let mut s = status.lock().await;
            s.error = Some(format_uv_sync_failure(root, out.status.code(), &stderr));
            return;
        }
        Err(e) => {
            let mut s = status.lock().await;
            s.error = Some(format_uv_sync_spawn_error(root, &uv_bin, &e.to_string()));
            return;
        }
        Ok(_) => {} // success — fall through
    }

    {
        let mut s = status.lock().await;
        s.detail = "Verifying Rust extension (openjarvis_rust)...".into();
    }
    if let Err(err) = verify_openjarvis_rust_extension(root, &uv_bin).await {
        let mut s = status.lock().await;
        s.error = Some(err);
        return;
    }

    {
        let mut s = status.lock().await;
        s.detail = format!("Starting API server from {}...", root.display());
    }

    let mut cmd = local_command(&uv_bin);
    let mut serve_argv: Vec<String> = vec![
        "run".into(),
        "jarvis".into(),
        "serve".into(),
        "--port".into(),
        JARVIS_PORT.to_string(),
    ];
    serve_argv.extend(plan.serve_args.iter().cloned());
    if use_cloudflare {
        serve_argv.extend(["--host".into(), "127.0.0.1".into()]);
    }
    // If the Ollama pull fell back to a different tag than planned, serve the
    // tag that is actually present. boot_plan always emits `--model` followed
    // immediately by its value, so `i + 1` is in bounds.
    if let Some(m) = &serve_model_override {
        match serve_argv.iter().position(|a| a == "--model") {
            Some(i) if i + 1 < serve_argv.len() => serve_argv[i + 1] = m.clone(),
            _ => eprintln!(
                "Warning: resolved model {:?} could not be applied; \
                 '--model <value>' not found in serve args {:?}",
                m, serve_argv
            ),
        }
    }
    cmd.args(&serve_argv)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .current_dir(root);
    // Avoid LD_LIBRARY_PATH leak when running inside an AppImage (#455) —
    // do this BEFORE cmd.env() calls below so our explicit cloud-key env
    // additions aren't accidentally stripped.
    prepare_subprocess_for_appimage(&mut cmd);
    prepare_backend_python(&mut cmd);

    if let Some(credentials) = &cloudflare_credentials {
        configure_cloudflare_environment(&mut cmd, credentials);
    } else {
        // Inject other provider keys from secure desktop storage.
        for (key, value) in cloud_keys_for_inference(&cfg) {
            cmd.env(&key, &value);
        }
    }
    if let Err(e) = spawn_managed_server(&backend, &mut cmd, !use_cloudflare).await {
        status.lock().await.error = Some(format!(
            "Could not start jarvis server: {}. \
             Make sure uv is installed (https://astral.sh/uv) and the OpenJarvis repo is cloned at {}",
            e,
            root.display(),
        ));
        return;
    }

    let server_url = format!("http://127.0.0.1:{}/health", JARVIS_PORT);
    let target = JarvisHealthTarget::Standard { url: &server_url, cloudflare: use_cloudflare };
    match wait_for_jarvis_health(target, Duration::from_secs(600), &backend).await {
        JarvisStartResult::Ready => {}
        _ if use_cloudflare => {
            status.lock().await.error = Some(
                "Cloudflare inference is unavailable. Check your Cloudflare credentials and connection.".into(),
            );
            return;
        }
        JarvisStartResult::IncompatiblePackagedApi => unreachable!("standard backend health check"),
        JarvisStartResult::ServiceUnavailable(body) => {
            let mut s = status.lock().await;
            s.error = Some(format!(
                "Jarvis server is running but the inference engine is not available \
                 (HTTP 503). This usually means the configured model couldn't be loaded.\n\n\
                 Check the server logs, or run 'uv run jarvis serve --port {}{}' \
                 from {} to see the engine error.\n\n\
                 Server response:\n{}",
                JARVIS_PORT,
                // Show the args actually passed (after `serve --port <port>`),
                // including any post-fallback `--model` override.
                match serve_argv.get(5..) {
                    Some(rest) if !rest.is_empty() => format!(" {}", rest.join(" ")),
                    _ => String::new(),
                },
                root.display(),
                body.trim(),
            ));
            return;
        }
        JarvisStartResult::EarlyExit { code, stderr } => {
            // `None` here means the OS didn't expose an exit code — on
            // Unix that's a signal kill (SIGKILL/SIGSEGV/...), on Windows
            // it means the process was terminated externally (Task
            // Manager, parent-of-parent, AV). "unknown" covers both.
            let code_str = code
                .map(|c| c.to_string())
                .unwrap_or_else(|| "unknown".into());
            let mut s = status.lock().await;
            s.error = Some(if stderr.is_empty() {
                format!(
                    "Jarvis server exited (code {}) before becoming ready.\n\n\
                     No stderr output. Check that:\n\
                     1. uv is installed ({})\n\
                     2. The OpenJarvis repo is at {}\n\
                     3. 'uv sync' completes in that directory",
                    code_str,
                    uv_bin,
                    root.display(),
                )
            } else {
                format!(
                    "Jarvis server exited (code {}) before becoming ready.\n\nStderr:\n{}",
                    code_str, stderr,
                )
            });
            return;
        }
        JarvisStartResult::Timeout => {
            let stderr = read_jarvis_stderr_tail(&backend).await;
            let mut s = status.lock().await;
            s.error = Some(if stderr.is_empty() {
                format!(
                    "Jarvis server did not become ready within 10 minutes. Check that:\n\
                     1. uv is installed ({})\n\
                     2. The OpenJarvis repo is at {}\n\
                     3. Run 'uv sync' in that directory",
                    uv_bin,
                    root.display(),
                )
            } else {
                format!(
                    "Jarvis server did not become ready within 10 minutes.\n\nStderr:\n{}",
                    stderr,
                )
            });
            return;
        }
    }

    if let Err(err) = pending_rollback.confirm(&mut cfg) {
        let mut s = status.lock().await;
        s.error = Some(format!("Could not confirm inference setup: {}", err));
        return;
    }

    {
        let mut s = status.lock().await;
        s.server_ready = true;
        s.phase = "ready".into();
        if use_cloudflare {
            s.model_ready = true;
            s.ollama_ready = true;
            s.detail = "Cloudflare inference ready.".into();
        } else {
            s.detail = "All systems ready.".into();
        }
    }

    // Phase 4: done. We intentionally do NOT auto-pull the rest of the
    // Qwen3.5 ladder here. The previous behavior walked every model that
    // "fit" in RAM (up to qwen3.5:122b ≈ 81 GB) and pulled each one in an
    // un-cancellable background task — so the app silently consumed tens of
    // gigabytes with no way to stop short of deleting it. The startup model
    // pulled in Phase 2 is enough to make the app fully usable; additional
    // models are now opt-in (Settings → "ollama pull <model>", or the
    // `pull_model` command invoked from the UI).
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

fn api_base() -> String {
    format!("http://127.0.0.1:{}", JARVIS_PORT)
}

#[tauri::command]
async fn get_setup_status(state: tauri::State<'_, SharedStatus>) -> Result<SetupStatus, String> {
    Ok(state.lock().await.clone())
}

#[tauri::command]
fn get_api_base() -> String {
    api_base()
}

#[tauri::command]
fn is_offline_distribution() -> bool {
    cfg!(feature = "windows-offline")
}

fn offline_source_allowed(cfg: &InferenceConfig) -> bool {
    match cfg.kind {
        SourceKind::Ollama => true,
        SourceKind::Cloudflare => validate_cloudflare_source(
            cfg.model.as_deref(),
            cfg.host.is_some(),
            cfg.engine.is_some(),
            false,
        ).is_ok(),
        SourceKind::Custom => offline::is_loopback_url(cfg.host.as_deref().unwrap_or(""))
            && matches!(
                cfg.engine.as_deref().unwrap_or(CUSTOM_FALLBACK_ENGINE),
                "lmstudio" | "llamacpp" | "ollama" | "vllm" | "sglang" | "lemonade" | "mlx"
            ),
    }
}

fn offline_serve_args(plan: &BootPlan, installed_model: Option<&str>) -> Vec<String> {
    let mut args = plan.serve_args.clone();
    if let Some(model) = installed_model {
        if let Some(index) = args.iter().position(|arg| arg == "--model") {
            args[index + 1] = model.to_owned();
        }
    }
    let host = if plan.launch_ollama {
        Some(format!("http://127.0.0.1:{OLLAMA_PORT}"))
    } else {
        plan.engine_host.as_ref().map(|(_, host)| host.clone())
    };
    if let Some(host) = host {
        args.extend(["--local-engine-host".into(), host]);
    }
    args
}

#[tauri::command]
async fn start_backend(
    backend: tauri::State<'_, SharedBackend>,
    status: tauri::State<'_, SharedStatus>,
) -> Result<(), String> {
    let Some(cfg) = read_configured_inference_config() else {
        return Err("Choose an inference source before starting OpenJarvis.".into());
    };
    if is_offline_distribution() {
        backend.lock().await.stop_all().await;
        stop_attached_backend(backend.inner(), JARVIS_PORT).await?;
    }
    *status.lock().await = SetupStatus::starting(&cfg);
    let b = backend.inner().clone();
    let s = status.inner().clone();
    start_managed_boot(b, s, true).await;
    Ok(())
}

#[tauri::command]
async fn stop_backend(backend: tauri::State<'_, SharedBackend>) -> Result<(), String> {
    backend.lock().await.stop_all().await;
    if is_offline_distribution() {
        stop_attached_backend(backend.inner(), JARVIS_PORT).await?;
    }
    Ok(())
}

/// Abort startup, stop every child, discard an unworkable source, and return
/// the UI to the inert chooser. This is the recovery path for setup failures.
#[tauri::command]
async fn reset_inference_source(
    backend: tauri::State<'_, SharedBackend>,
    status: tauri::State<'_, SharedStatus>,
) -> Result<(), String> {
    backend.lock().await.stop_all().await;
    if is_offline_distribution() {
        stop_attached_backend(backend.inner(), JARVIS_PORT).await?;
    }
    discard_pending_inference_setup()?;
    *status.lock().await = SetupStatus::default();
    if is_offline_distribution() {
        start_managed_boot(backend.inner().clone(), status.inner().clone(), false).await;
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum PackagedBackendHealth {
    Inference,
    InferenceUnavailable,
    LocalServices,
    Unreachable,
}

fn classify_packaged_health(
    status: reqwest::StatusCode,
    body: &serde_json::Value,
) -> PackagedBackendHealth {
    use PackagedBackendHealth::*;

    match status {
        reqwest::StatusCode::SERVICE_UNAVAILABLE if body["detail"] == "Engine unhealthy" => {
            InferenceUnavailable
        }
        reqwest::StatusCode::OK
            if body["status"] == "ok"
                && body["mode"] == "local_services_only"
                && body["inference_available"] == false =>
        {
            LocalServices
        }
        reqwest::StatusCode::OK if body["status"] == "ok" && body.get("mode").is_none() => {
            Inference
        }
        _ => Unreachable,
    }
}

async fn owned_server_pid(backend: &SharedBackend) -> Option<u32> {
    let mut manager = backend.lock().await;
    let child = &mut manager.jarvis.as_mut()?.child;
    if child.try_wait().ok()?.is_some() {
        return None;
    }
    child.id()
}

// Keep in sync with the revision served by openjarvis.server.daemon.
const PACKAGED_BACKEND_API_HEADER: &str = "x-ace-backend-api";
const PACKAGED_BACKEND_API_REVISION: &str = "2";

#[derive(Debug, PartialEq, Eq)]
enum PackagedHealthProbe {
    Compatible(PackagedBackendHealth),
    Incompatible,
}

fn matches_packaged_api(headers: &reqwest::header::HeaderMap) -> bool {
    let mut values = headers.get_all(PACKAGED_BACKEND_API_HEADER).iter();
    values.next().is_some_and(|value| value == PACKAGED_BACKEND_API_REVISION)
        && values.next().is_none()
}

struct PackagedHealthResponse {
    health: PackagedBackendHealth,
    compatible_api: bool,
    cloudflare: bool,
}

async fn probe_packaged_health(port: u16) -> Option<PackagedHealthResponse> {
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(5))
        .build()
        .ok()?;
    let response = client.get(format!("http://127.0.0.1:{port}/health")).send().await.ok()?;
    let status = response.status();
    let compatible_api = matches_packaged_api(response.headers());
    let cloudflare = cloudflare_response(response.headers());
    let body = response.json().await.ok()?;
    Some(PackagedHealthResponse {
        health: classify_packaged_health(status, &body),
        compatible_api,
        cloudflare,
    })
}

async fn find_existing_bundled_server(
    runtime: &offline::BundledBackend,
    port: u16,
    expected_pid: Option<u32>,
) -> Option<(offline::AttachedBackend, PackagedHealthProbe)> {
    let identity = runtime.identify_server(port).await?;
    if expected_pid.is_some_and(|pid| identity.pid != pid) {
        return None;
    }
    let probe = probe_verified_bundled_server(runtime, port, &identity).await?;
    Some((identity, probe))
}

async fn probe_verified_bundled_server(
    runtime: &offline::BundledBackend,
    port: u16,
    identity: &offline::AttachedBackend,
) -> Option<PackagedHealthProbe> {
    let probe = classify_verified_packaged_health(identity.mode, probe_packaged_health(port).await?)?;
    (runtime.identify_server(port).await.as_ref() == Some(identity)).then_some(probe)
}

fn classify_verified_packaged_health(
    mode: offline::AttachedBackendMode,
    response: PackagedHealthResponse,
) -> Option<PackagedHealthProbe> {
    use offline::AttachedBackendMode::{Cloudflare, Inference, LocalServices};
    use PackagedBackendHealth as Health;

    if !matches!(
        (mode, &response.health),
        (LocalServices, Health::LocalServices)
            | (Inference | Cloudflare, Health::Inference | Health::InferenceUnavailable)
    ) {
        return None;
    }
    if !response.compatible_api {
        return Some(PackagedHealthProbe::Incompatible);
    }
    if response.cloudflare != (mode == Cloudflare) {
        return None;
    }
    Some(PackagedHealthProbe::Compatible(response.health))
}

async fn probe_owned_packaged_health(
    port: u16,
    pid: u32,
    backend: &SharedBackend,
) -> Option<PackagedHealthResponse> {
    #[cfg(target_os = "windows")]
    if !windows_tcp::owns_loopback_listener(port, pid) {
        return None;
    }
    let response = probe_packaged_health(port).await?;
    #[cfg(target_os = "windows")]
    if !windows_tcp::owns_loopback_listener(port, pid) {
        return None;
    }
    (owned_server_pid(backend).await == Some(pid)).then_some(response)
}

async fn check_packaged_start_health(
    _runtime: &offline::BundledBackend,
    port: u16,
    mode: offline::AttachedBackendMode,
    backend: &SharedBackend,
) -> Option<JarvisStartResult> {
    let pid = owned_server_pid(backend).await?;
    #[cfg(target_os = "windows")]
    let probe = classify_verified_packaged_health(mode, probe_owned_packaged_health(port, pid, backend).await?)?;
    #[cfg(not(target_os = "windows"))]
    let probe = {
        let (identity, probe) = find_existing_bundled_server(_runtime, port, Some(pid)).await?;
        if identity.mode != mode || owned_server_pid(backend).await != Some(pid) {
            return None;
        }
        probe
    };
    match probe {
        PackagedHealthProbe::Compatible(
            PackagedBackendHealth::Inference | PackagedBackendHealth::LocalServices,
        ) => Some(JarvisStartResult::Ready),
        PackagedHealthProbe::Compatible(PackagedBackendHealth::InferenceUnavailable) => {
            Some(JarvisStartResult::ServiceUnavailable("Engine unhealthy".into()))
        }
        PackagedHealthProbe::Incompatible => Some(JarvisStartResult::IncompatiblePackagedApi),
        PackagedHealthProbe::Compatible(PackagedBackendHealth::Unreachable) => None,
    }
}

fn may_attach_bundled_mode(
    mode: offline::AttachedBackendMode,
    cloudflare_model: Option<&str>,
    config: Option<&InferenceConfig>,
    try_inference: bool,
) -> bool {
    use offline::AttachedBackendMode::{Cloudflare, Inference, LocalServices};

    match mode {
        LocalServices => true,
        Inference => try_inference && !config.is_some_and(|cfg| cfg.kind == SourceKind::Cloudflare),
        Cloudflare => try_inference
            && config.is_some_and(|cfg| {
                cfg.kind == SourceKind::Cloudflare && cfg.confirmed
                    && validate_cloudflare_source(
                        cfg.model.as_deref(),
                        cfg.host.is_some(),
                        cfg.engine.is_some(),
                        false,
                    ).is_ok_and(|model| Some(model) == cloudflare_model)
            }),
    }
}

fn attached_recovery_status(error: &str) -> SetupStatus {
    SetupStatus {
        phase: "server".into(),
        detail: "Waiting to restart ACE local services.".into(),
        error: Some(error.into()),
        requires_source: false,
        ..SetupStatus::default()
    }
}

async fn try_attach_bundled_server(
    runtime: &offline::BundledBackend,
    port: u16,
    backend: &SharedBackend,
    status: &SharedStatus,
    try_inference: bool,
) -> bool {
    let Some(identity) = runtime.identify_server(port).await else {
        return false;
    };
    let config = read_configured_inference_config();
    let next_status = if may_attach_bundled_mode(
        identity.mode, identity.cloudflare_model.as_deref(), config.as_ref(), try_inference,
    ) {
        match probe_verified_bundled_server(runtime, port, &identity).await {
            Some(PackagedHealthProbe::Compatible(health)) => attached_setup_status(health),
            Some(PackagedHealthProbe::Incompatible) => attached_recovery_status(
                "The running ACE backend uses a different API version. Choose Return to local services to restart the backend for this version of ACE.",
            ),
            None => return false,
        }
    } else {
        // Keep the identified process available for explicit recovery without
        // probing a cloud inference backend from a local startup path.
        attached_recovery_status(
            "The running ACE backend uses a different inference source or model. Choose Return to local services to restart it, then choose the source to use.",
        )
    };
    backend.lock().await.attached = Some(identity);
    *status.lock().await = next_status;
    true
}

fn attached_setup_status(health: PackagedBackendHealth) -> SetupStatus {
    let model_ready = health == PackagedBackendHealth::Inference;
    let source = read_configured_inference_config()
        .map_or("unconfigured", |cfg| cfg.kind.label());
    let detail = match health {
        PackagedBackendHealth::Inference => "Connected to the ACE backend already running.",
        PackagedBackendHealth::LocalServices => "Connected to ACE local services already running.",
        _ => "Connected to the running ACE backend. Its inference source is unavailable.",
    };
    SetupStatus {
        phase: if model_ready { "ready" } else { "local_services_ready" }.into(),
        detail: detail.into(),
        ollama_ready: model_ready,
        server_ready: true,
        model_ready,
        error: None,
        source: source.into(),
        requires_source: false,
    }
}

/// Only an explicit stop or configuration change may stop an adopted process.
async fn stop_attached_backend(backend: &SharedBackend, port: u16) -> Result<(), String> {
    let (root, attached) = {
        let manager = backend.lock().await;
        (manager.bundled_root.clone(), manager.attached.clone())
    };
    if let (Some(root), Some(attached)) = (root, attached) {
        offline::BundledBackend::open(&root)?
            .stop_attached_server(port, &attached)
            .await?;
        let mut manager = backend.lock().await;
        if manager.attached.as_ref() == Some(&attached) {
            manager.attached = None;
        }
    }
    Ok(())
}

/// Probe only a server started or independently verified by this desktop.
#[tauri::command]
async fn check_packaged_backend_health(
    backend: tauri::State<'_, SharedBackend>,
) -> Result<PackagedBackendHealth, String> {
    use PackagedBackendHealth::Unreachable;

    if !is_offline_distribution() {
        return Ok(Unreachable);
    }
    if let Some(pid) = owned_server_pid(backend.inner()).await {
        if let Some(response) = probe_owned_packaged_health(JARVIS_PORT, pid, backend.inner()).await {
            if response.compatible_api {
                return Ok(response.health);
            }
        }
        return Ok(Unreachable);
    }
    let (root, attached) = {
        let manager = backend.lock().await;
        (manager.bundled_root.clone(), manager.attached.clone())
    };
    if let (Some(root), Some(expected)) = (root, attached) {
        if !may_attach_bundled_mode(
            expected.mode,
            expected.cloudflare_model.as_deref(),
            read_configured_inference_config().as_ref(),
            true,
        ) {
            return Ok(Unreachable);
        }
        if let Ok(runtime) = offline::BundledBackend::open(&root) {
            if let Some((identity, PackagedHealthProbe::Compatible(health))) =
                find_existing_bundled_server(&runtime, JARVIS_PORT, Some(expected.pid)).await
            {
                if identity == expected && backend.lock().await.attached.as_ref() == Some(&identity) {
                    return Ok(health);
                }
            }
        }
    }
    Ok(Unreachable)
}

#[tauri::command]
async fn check_health(api_url: String) -> Result<serde_json::Value, String> {
    let url = format!(
        "{}/health",
        if api_url.is_empty() {
            api_base()
        } else {
            api_url
        }
    );
    let resp = reqwest::get(&url)
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    resp.json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))
}

#[tauri::command]
async fn fetch_energy(api_url: String) -> Result<serde_json::Value, String> {
    let base = if api_url.is_empty() {
        api_base()
    } else {
        api_url
    };
    let resp = reqwest::get(format!("{}/v1/telemetry/energy", base))
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    resp.json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))
}

#[tauri::command]
async fn fetch_telemetry(api_url: String) -> Result<serde_json::Value, String> {
    let base = if api_url.is_empty() {
        api_base()
    } else {
        api_url
    };
    let resp = reqwest::get(format!("{}/v1/telemetry/stats", base))
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    resp.json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))
}

#[tauri::command]
async fn fetch_traces(api_url: String, limit: u32) -> Result<serde_json::Value, String> {
    let base = if api_url.is_empty() {
        api_base()
    } else {
        api_url
    };
    let resp = reqwest::get(format!("{}/v1/traces?limit={}", base, limit))
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    resp.json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))
}

#[tauri::command]
async fn fetch_trace(api_url: String, trace_id: String) -> Result<serde_json::Value, String> {
    let base = if api_url.is_empty() {
        api_base()
    } else {
        api_url
    };
    let resp = reqwest::get(format!("{}/v1/traces/{}", base, trace_id))
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    resp.json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))
}

#[tauri::command]
async fn fetch_learning_stats(api_url: String) -> Result<serde_json::Value, String> {
    let base = if api_url.is_empty() {
        api_base()
    } else {
        api_url
    };
    let resp = reqwest::get(format!("{}/v1/learning/stats", base))
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    resp.json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))
}

#[tauri::command]
async fn fetch_learning_policy(api_url: String) -> Result<serde_json::Value, String> {
    let base = if api_url.is_empty() {
        api_base()
    } else {
        api_url
    };
    let resp = reqwest::get(format!("{}/v1/learning/policy", base))
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    resp.json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))
}

#[tauri::command]
async fn fetch_memory_stats(api_url: String) -> Result<serde_json::Value, String> {
    let base = if api_url.is_empty() {
        api_base()
    } else {
        api_url
    };
    let resp = reqwest::get(format!("{}/v1/memory/stats", base))
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    resp.json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))
}

#[tauri::command]
async fn search_memory(
    api_url: String,
    query: String,
    top_k: u32,
) -> Result<serde_json::Value, String> {
    let base = if api_url.is_empty() {
        api_base()
    } else {
        api_url
    };
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/memory/search", base))
        .json(&serde_json::json!({"query": query, "top_k": top_k}))
        .send()
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    resp.json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))
}

#[tauri::command]
async fn fetch_agents(api_url: String) -> Result<serde_json::Value, String> {
    let base = if api_url.is_empty() {
        api_base()
    } else {
        api_url
    };
    let resp = reqwest::get(format!("{}/v1/agents", base))
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    resp.json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))
}

#[tauri::command]
async fn fetch_models(api_url: String) -> Result<serde_json::Value, String> {
    let base = if api_url.is_empty() {
        api_base()
    } else {
        api_url
    };
    let resp = reqwest::get(format!("{}/v1/models", base))
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    resp.json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))
}

#[tauri::command]
async fn run_jarvis_command(
    args: Vec<String>,
    backend: tauri::State<'_, SharedBackend>,
    status: tauri::State<'_, SharedStatus>,
) -> Result<String, String> {
    if is_offline_distribution() {
        match args.first().map(String::as_str) {
            Some("serve") => {
                backend.lock().await.stop_all().await;
                stop_attached_backend(backend.inner(), JARVIS_PORT).await?;
                start_managed_boot(backend.inner().clone(), status.inner().clone(), true).await;
                return Ok("Starting the packaged ACE backend.".into());
            }
            Some("stop") => {
                backend.lock().await.stop_all().await;
                stop_attached_backend(backend.inner(), JARVIS_PORT).await?;
                *status.lock().await = SetupStatus::default();
                return Ok("The packaged ACE backend has stopped.".into());
            }
            _ => return Err("Unsupported backend management command.".into()),
        }
    }
    let uv_bin = resolve_bin("uv");

    let mut cmd_args = vec!["run".to_string(), "jarvis".to_string()];
    cmd_args.extend(args.iter().cloned());

    let mut cmd = local_command(&uv_bin);
    cmd.args(&cmd_args);
    // Run from the project root so `uv run jarvis` resolves the OpenJarvis
    // project regardless of the app's launch cwd. In a packaged install the
    // cwd isn't the checkout, so without this `jarvis` isn't found and the
    // backend never starts — the UI then shows "Failed to get response"
    // (see #531).
    if let Some(ref root) = find_project_root() {
        cmd.current_dir(root);
    }

    let is_serve = args.first().map(|a| a.as_str() == "serve").unwrap_or(false);

    if !is_serve {
        // Short-lived command (e.g. `stop`, `status`): wait for it and return
        // its captured output.
        let output = cmd
            .output()
            .await
            .map_err(|e| format!("Failed to launch jarvis: {}", e))?;
        return if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).to_string())
        };
    }

    // `jarvis serve` is a long-running server that never exits. The old code
    // used `.output()`, which waits for the process to exit and so hung this
    // command forever — the "Start" button never resolved (#531). Spawn it
    // detached instead, drain stderr (a full 4 KB Windows pipe can otherwise
    // stall the child mid-startup, #309), and poll /health for readiness.
    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to launch jarvis serve: {}", e))?;

    let tail: StderrTail = Arc::new(Mutex::new(Vec::new()));
    if let Some(stderr) = child.stderr.take() {
        spawn_jarvis_stderr_drainer(stderr, tail.clone());
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))?;
    let url = format!("http://127.0.0.1:{}/health", JARVIS_PORT);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);

    loop {
        // Surface an early crash (bad venv, missing Rust ext, etc.) right away
        // instead of waiting out the full readiness timeout.
        if let Ok(Some(status)) = child.try_wait() {
            let stderr = String::from_utf8_lossy(tail.lock().await.as_slice()).into_owned();
            return Err(format!(
                "jarvis serve exited (code {:?}) before becoming healthy:\n{}",
                status.code(),
                stderr.trim()
            ));
        }
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                // Leave the server running (the Child is detached on drop —
                // kill_on_drop defaults to false); `stop` tears it down.
                return Ok(format!(
                    "jarvis serve is ready on http://127.0.0.1:{}",
                    JARVIS_PORT
                ));
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(format!(
                "jarvis serve did not become healthy on port {} within 120s.",
                JARVIS_PORT
            ));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

#[tauri::command]
async fn fetch_savings(api_url: String) -> Result<serde_json::Value, String> {
    let base = if api_url.is_empty() {
        api_base()
    } else {
        api_url
    };
    let resp = reqwest::get(format!("{}/v1/savings", base))
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    resp.json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))
}

/// Transcribe audio via the speech API endpoint.
#[tauri::command]
async fn transcribe_audio(
    api_url: String,
    audio_data: Vec<u8>,
    filename: String,
) -> Result<serde_json::Value, String> {
    let url = format!("{}/v1/speech/transcribe", api_url);
    let client = reqwest::Client::new();

    let part = reqwest::multipart::Part::bytes(audio_data)
        .file_name(filename)
        .mime_str("audio/webm")
        .map_err(|e| format!("Failed to create multipart: {}", e))?;

    let form = reqwest::multipart::Form::new().part("file", part);

    let resp = client
        .post(&url)
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    let status = resp.status();
    let body = resp
        .text()
        .await
        .map_err(|e| format!("Invalid response: {}", e))?;
    if !status.is_success() {
        let detail = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|value| {
                value
                    .get("detail")
                    .and_then(|detail| detail.as_str())
                    .map(str::to_string)
            })
            .filter(|detail| !detail.is_empty())
            .unwrap_or(body);
        return Err(format!(
            "Transcription failed ({}): {}",
            status.as_u16(),
            detail
        ));
    }
    serde_json::from_str(&body).map_err(|e| format!("Invalid response: {}", e))
}

/// Submit savings to Supabase leaderboard.
#[tauri::command]
async fn submit_savings(
    supabase_url: String,
    supabase_key: String,
    payload: serde_json::Value,
) -> Result<bool, String> {
    if supabase_url.is_empty() || supabase_key.is_empty() {
        return Ok(false);
    }
    let client = reqwest::Client::new();
    let resp = client
        .post(format!(
            "{}/rest/v1/savings_entries?on_conflict=anon_id",
            supabase_url
        ))
        .header("Content-Type", "application/json")
        .header("apikey", &supabase_key)
        .header("Authorization", format!("Bearer {}", supabase_key))
        .header("Prefer", "resolution=merge-duplicates")
        .json(&payload)
        .send()
        .await
        .map_err(|e| format!("Supabase POST failed: {}", e))?;
    Ok(resp.status().is_success())
}

// ---------------------------------------------------------------------------
// Cloud API key management
// ---------------------------------------------------------------------------

const SECURE_KEY_SERVICE: &str = "OpenJarvis Cloud Keys";
// A first-run custom credential is kept in its own secure-storage slot until
// the managed backend has passed every readiness gate.  In particular, do not
// overwrite an existing <ENGINE>_API_KEY while setup can still be cancelled.
const PENDING_INFERENCE_API_KEY: &str = "OPENJARVIS_PENDING_INFERENCE_API_KEY";
const MANAGED_CLOUD_KEY_NAMES: &[&str] = &[
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "ATLASCLOUD_API_KEY",
    "MINIMAX_API_KEY",
    "TAVILY_API_KEY",
];

/// Legacy path used by older desktop builds. New saves never write here.
fn legacy_cloud_keys_path() -> std::path::PathBuf {
    let home = home_dir();
    std::path::PathBuf::from(home)
        .join(".openjarvis")
        .join("cloud-keys.env")
}

fn validate_cloud_key_name(key_name: &str) -> Result<(), String> {
    let valid = !key_name.is_empty()
        && key_name.len() <= 128
        && key_name.ends_with("_API_KEY")
        && key_name
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_');
    if valid {
        Ok(())
    } else {
        Err(format!("Invalid API key name: {}", key_name))
    }
}

fn engine_api_key_name(engine: &str) -> String {
    let normalized: String = engine
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = normalized.trim_matches('_');
    let engine_name = if trimmed.is_empty() {
        CUSTOM_FALLBACK_ENGINE.to_ascii_uppercase()
    } else {
        trimmed.to_string()
    };
    format!("{}_API_KEY", engine_name)
}

fn managed_cloud_key_names() -> Vec<String> {
    let mut names: Vec<String> = MANAGED_CLOUD_KEY_NAMES
        .iter()
        .map(|name| (*name).to_string())
        .collect();

    let cfg = read_inference_config();
    if matches!(&cfg.kind, SourceKind::Custom) {
        let engine = cfg
            .engine
            .unwrap_or_else(|| CUSTOM_FALLBACK_ENGINE.to_string());
        let key_name = engine_api_key_name(&engine);
        if validate_cloud_key_name(&key_name).is_ok() {
            names.push(key_name);
        }
    }

    names.sort();
    names.dedup();
    names
}

fn secure_store_get(key_name: &str) -> Result<Option<String>, String> {
    validate_cloud_key_name(key_name)?;
    let entry = keyring::Entry::new(SECURE_KEY_SERVICE, key_name).map_err(|err| {
        format!(
            "Failed to open secure key storage for {}: {}",
            key_name, err
        )
    })?;
    match entry.get_password() {
        Ok(value) => Ok(Some(value)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(err) => Err(format!(
            "Failed to read {} from secure key storage: {}",
            key_name, err
        )),
    }
}

fn secure_store_set(key_name: &str, key_value: &str) -> Result<(), String> {
    validate_cloud_key_name(key_name)?;
    let entry = keyring::Entry::new(SECURE_KEY_SERVICE, key_name).map_err(|err| {
        format!(
            "Failed to open secure key storage for {}: {}",
            key_name, err
        )
    })?;
    if key_value.is_empty() {
        return match entry.delete_credential() {
            Ok(()) => Ok(()),
            Err(keyring::Error::NoEntry) => Ok(()),
            Err(err) => Err(format!(
                "Failed to remove {} from secure key storage: {}",
                key_name, err
            )),
        };
    }
    entry
        .set_password(key_value)
        .map_err(|err| format!("Failed to save {} in secure key storage: {}", key_name, err))
}

trait InferenceCredentialStore: Send + Sync {
    fn get(&self, key_name: &str) -> Result<Option<String>, String>;
    fn set(&self, key_name: &str, key_value: &str) -> Result<(), String>;
}

struct SystemInferenceCredentialStore;

impl InferenceCredentialStore for SystemInferenceCredentialStore {
    fn get(&self, key_name: &str) -> Result<Option<String>, String> {
        secure_store_get(key_name)
    }

    fn set(&self, key_name: &str, key_value: &str) -> Result<(), String> {
        secure_store_set(key_name, key_value)
    }
}

fn read_legacy_cloud_keys() -> Vec<(String, String)> {
    let path = legacy_cloud_keys_path();
    let mut keys = Vec::new();
    if let Ok(contents) = std::fs::read_to_string(&path) {
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                keys.push((k.trim().to_string(), v.trim().to_string()));
            }
        }
    }
    keys
}

fn migrate_legacy_cloud_keys() {
    let path = legacy_cloud_keys_path();
    if !path.exists() {
        return;
    }

    let legacy_keys = read_legacy_cloud_keys();
    if legacy_keys.is_empty() {
        let _ = std::fs::remove_file(&path);
        return;
    }

    let mut migrated_all = true;
    for (key, value) in legacy_keys {
        if value.is_empty() {
            continue;
        }
        if secure_store_set(&key, &value).is_err() {
            migrated_all = false;
        }
    }

    if migrated_all {
        let _ = std::fs::remove_file(path);
    }
}

/// Read cloud keys from secure desktop storage and return key=value pairs.
fn read_cloud_keys() -> Vec<(String, String)> {
    migrate_legacy_cloud_keys();
    managed_cloud_key_names()
        .into_iter()
        .filter_map(|key| match secure_store_get(&key) {
            Ok(Some(value)) if !value.is_empty() => Some((key, value)),
            _ => None,
        })
        .collect()
}

/// Overlay a staged first-run credential onto the child-process environment.
/// The durable provider key remains untouched until setup is confirmed.
fn cloud_keys_for_inference(cfg: &InferenceConfig) -> Vec<(String, String)> {
    let mut keys = read_cloud_keys();
    let Some(credential) = PendingInferenceCredential::for_config(cfg) else {
        return keys;
    };
    let Ok(Some(staged)) = secure_store_get(PENDING_INFERENCE_API_KEY) else {
        return keys;
    };
    if staged.is_empty() {
        return keys;
    }
    keys.retain(|(name, _)| name != &credential.target_key);
    keys.push((credential.target_key, staged));
    keys
}

async fn reload_cloud_keys_at(reload_url: &str, keys: Vec<(String, String)>) {
    let key_map: serde_json::Map<String, serde_json::Value> = keys
        .into_iter()
        .map(|(key, value)| (key, serde_json::Value::String(value)))
        .collect();
    let _ = reqwest::Client::new()
        .post(reload_url)
        .json(&serde_json::json!({ "keys": key_map }))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await;
}

/// Hot-reload only a live server process launched and owned by this desktop
/// process.  A healthy-looking loopback listener is not proof of ownership and
/// must never receive raw credentials.
async fn reload_cloud_keys_for_owned_backend(
    backend: &SharedBackend,
    status: &SharedStatus,
    keys: Vec<(String, String)>,
) {
    let reload_url = format!("http://127.0.0.1:{}/v1/cloud/reload", JARVIS_PORT);
    reload_cloud_keys_for_owned_backend_at(backend, status, keys, &reload_url).await;
}

async fn reload_cloud_keys_for_owned_backend_at(
    backend: &SharedBackend,
    status: &SharedStatus,
    keys: Vec<(String, String)>,
    reload_url: &str,
) {
    if !status.lock().await.server_ready {
        return;
    }

    // Keep the manager lock for the request so stop/reset cannot kill our
    // child and let an unrelated process claim the port between the ownership
    // check and the credential POST.
    let mut manager = backend.lock().await;
    let Some(handle) = manager.jarvis.as_mut() else {
        return;
    };
    if !matches!(handle.child.try_wait(), Ok(None)) {
        return;
    }
    reload_cloud_keys_at(reload_url, keys).await;
}

/// Save a single cloud API key to secure desktop storage.
#[tauri::command]
async fn save_cloud_key(
    key_name: String,
    key_value: String,
    backend: tauri::State<'_, SharedBackend>,
    status: tauri::State<'_, SharedStatus>,
) -> Result<(), String> {
    if is_offline_distribution() {
        return Err("Cloud API key setup is available in the standard desktop edition.".into());
    }
    let key_value = key_value.trim().to_string();
    secure_store_set(&key_name, &key_value)?;

    // Tell only our own ready child to hot-reload its cloud engine.  Posting to
    // a merely responsive port would disclose the raw key to a port squatter.
    reload_cloud_keys_for_owned_backend(
        backend.inner(),
        status.inner(),
        vec![(key_name, key_value)],
    )
    .await;

    Ok(())
}

/// Get which cloud providers have keys configured (without exposing values).
#[tauri::command]
async fn get_cloud_key_status() -> Result<serde_json::Value, String> {
    if is_offline_distribution() {
        return Ok(serde_json::json!([]));
    }
    migrate_legacy_cloud_keys();
    let status: Vec<serde_json::Value> = managed_cloud_key_names()
        .into_iter()
        .map(|key| {
            let set = matches!(secure_store_get(&key), Ok(Some(value)) if !value.is_empty());
            serde_json::json!({ "key": key, "set": set })
        })
        .collect();
    Ok(serde_json::json!(status))
}

/// Return the current inference-source config for the Settings UI.
#[tauri::command]
async fn get_inference_source() -> Result<InferenceConfig, String> {
    Ok(read_inference_config())
}

/// Persist the chosen inference source. `host` is normalized to a bare base
/// URL. For custom endpoints, an optional API key is stored in secure desktop
/// storage under `<ENGINE>_API_KEY`. Applies on next app launch.
#[tauri::command]
async fn set_inference_source(
    kind: String,
    model: Option<String>,
    host: Option<String>,
    engine: Option<String>,
    api_key: Option<String>,
    pending: Option<bool>,
) -> Result<(), String> {
    let kind = match kind.as_str() {
        "ollama" => SourceKind::Ollama,
        "custom" => SourceKind::Custom,
        "cloudflare" => SourceKind::Cloudflare,
        other => return Err(format!("Unknown inference source kind: {:?}", other)),
    };
    if kind == SourceKind::Custom && selects_cloudflare_engine(engine.as_deref()) {
        return Err("Select the Cloudflare inference source to use Workers AI.".into());
    }
    let model = match kind {
        SourceKind::Cloudflare => {
            Some(validate_cloudflare_source(
                model.as_deref(),
                host.is_some(),
                engine.is_some(),
                api_key.is_some(),
            )?.into())
        }
        SourceKind::Ollama => Some(normalize_ollama_model_tag(model.as_deref().unwrap_or(""))?),
        SourceKind::Custom => model
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
    };
    let cfg = InferenceConfig {
        kind,
        confirmed: !pending.unwrap_or(false),
        model,
        host: host.map(|h| normalize_host(&h)).filter(|h| !h.is_empty()),
        engine: engine.filter(|e| !e.is_empty()),
    };
    let api_key = api_key
        .map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty());
    if let SourceKind::Custom = cfg.kind {
        if cfg.host.is_none() {
            return Err("A server URL is required for a custom endpoint.".into());
        }
        if cfg.model.as_deref().unwrap_or("").is_empty() {
            return Err("A model name is required for a custom endpoint.".into());
        }
    }

    if is_offline_distribution() {
        if !offline_source_allowed(&cfg) || api_key.is_some() {
            return Err(
                "Choose a supported local engine or the Cloudflare Workers AI option.".into(),
            );
        }
        return write_inference_config_at(&cfg, &inference_config_path());
    }
    let store = SystemInferenceCredentialStore;
    if cfg.confirmed {
        persist_confirmed_inference_config(&cfg, api_key.as_deref(), &store)
    } else {
        stage_pending_inference_config(&cfg, api_key.as_deref(), &store)
    }
}

/// List installed models through the native local Ollama connection.
#[tauri::command]
async fn list_ollama_models() -> Result<Vec<serde_json::Value>, String> {
    fetch_ollama_models().await
}

/// Pull a model via Ollama (called from frontend download button).
#[tauri::command]
async fn pull_ollama_model(model_name: String) -> Result<serde_json::Value, String> {
    let model_name = normalize_ollama_model_tag(&model_name)?;
    pull_model(&model_name)
        .await
        .map_err(|e| format!("Failed to pull {}: {}", model_name, e))?;
    Ok(serde_json::json!({"status": "ok", "model": model_name}))
}

/// Delete a model from Ollama.
#[tauri::command]
async fn delete_ollama_model(model_name: String) -> Result<serde_json::Value, String> {
    let model_name = normalize_ollama_model_tag(&model_name)?;
    let url = format!("http://127.0.0.1:{}/api/delete", OLLAMA_PORT);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .delete(&url)
        .json(&serde_json::json!({"name": model_name}))
        .send()
        .await
        .map_err(|e| format!("Delete failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("Delete returned status {}", resp.status()));
    }
    Ok(serde_json::json!({"status": "deleted", "model": model_name}))
}

// ---------------------------------------------------------------------------
// Inference-source selection (~/.openjarvis/inference.json)
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum SourceKind {
    Ollama,
    Custom,
    Cloudflare,
}

impl SourceKind {
    fn label(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::Custom => "custom",
            Self::Cloudflare => "cloudflare",
        }
    }
}

fn selects_cloudflare_engine(engine: Option<&str>) -> bool {
    engine.is_some_and(|value| value.trim().eq_ignore_ascii_case("cloudflare"))
}

fn validate_cloudflare_source(
    model: Option<&str>,
    has_host: bool,
    has_engine: bool,
    has_api_key: bool,
) -> Result<&str, String> {
    if has_host || has_engine || has_api_key {
        return Err("Cloudflare Workers AI uses its own endpoint and credential settings.".into());
    }
    let model = model.map_or(CLOUDFLARE_MODEL, str::trim);
    let valid_segment = |part: &str| {
        part.as_bytes().first().is_some_and(u8::is_ascii_alphanumeric)
            && part.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
    };
    let mut parts = model.split('/');
    if model.len() <= 128
        && matches!(parts.next(), Some("@cf" | "@hf"))
        && parts.next().is_some_and(valid_segment)
        && parts.next().is_some_and(valid_segment)
        && parts.next().is_none()
    {
        return Ok(model);
    }
    Err("Enter a Cloudflare model ID in the form @cf/vendor/model or @hf/vendor/model.".into())
}

impl Default for SourceKind {
    fn default() -> Self {
        SourceKind::Ollama
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, Default)]
struct InferenceConfig {
    #[serde(default)]
    kind: SourceKind,
    /// First-run choices stay pending until the complete backend reaches
    /// ready. Legacy configs predate this field and remain trusted.
    #[serde(default = "legacy_config_is_confirmed")]
    confirmed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    /// Bare base URL (no trailing `/v1`), custom only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    host: Option<String>,
    /// OpenAI-compatible engine key (e.g. "lmstudio"), custom only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    engine: Option<String>,
}

fn legacy_config_is_confirmed() -> bool {
    true
}

/// Path to the inference-source config (~/.openjarvis/inference.json).
fn inference_config_path() -> std::path::PathBuf {
    if is_offline_distribution() {
        offline::profile_data_dir().join("inference-offline.json")
    } else {
        std::path::PathBuf::from(home_dir())
            .join(".openjarvis")
            .join("inference.json")
    }
}

fn remove_inference_config_at(path: &std::path::Path) -> Result<(), String> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(format!("Failed to clear inference config: {}", err)),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PendingInferenceCredential {
    target_key: String,
}

impl PendingInferenceCredential {
    fn for_config(cfg: &InferenceConfig) -> Option<Self> {
        if cfg.confirmed || !matches!(cfg.kind, SourceKind::Custom) {
            return None;
        }
        let engine = cfg.engine.as_deref().unwrap_or(CUSTOM_FALLBACK_ENGINE);
        Some(Self {
            target_key: engine_api_key_name(engine),
        })
    }
}

fn restore_credential(
    store: &dyn InferenceCredentialStore,
    key_name: &str,
    previous: Option<&str>,
) -> Result<(), String> {
    store.set(key_name, previous.unwrap_or(""))
}

/// Persist a returning-user source and its optional credential as one logical
/// transaction. If the config write fails, restore the exact prior key.
fn persist_confirmed_inference_config(
    cfg: &InferenceConfig,
    api_key: Option<&str>,
    store: &dyn InferenceCredentialStore,
) -> Result<(), String> {
    persist_confirmed_inference_config_at(cfg, api_key, store, inference_config_path().as_path())
}

fn persist_confirmed_inference_config_at(
    cfg: &InferenceConfig,
    api_key: Option<&str>,
    store: &dyn InferenceCredentialStore,
    path: &std::path::Path,
) -> Result<(), String> {
    let Some(api_key) = api_key else {
        return write_inference_config_at(cfg, path);
    };
    let Some(target_key) = (matches!(cfg.kind, SourceKind::Custom))
        .then(|| engine_api_key_name(cfg.engine.as_deref().unwrap_or(CUSTOM_FALLBACK_ENGINE)))
    else {
        return write_inference_config_at(cfg, path);
    };

    let previous = store
        .get(&target_key)
        .map_err(|err| format!("Could not read the existing API key: {}", err))?;
    store
        .set(&target_key, api_key)
        .map_err(|err| format!("Could not store the API key: {}", err))?;
    if let Err(config_err) = write_inference_config_at(cfg, path) {
        return match restore_credential(store, &target_key, previous.as_deref()) {
            Ok(()) => Err(config_err),
            Err(restore_err) => Err(format!(
                "{}; additionally could not restore the previous API key: {}",
                config_err, restore_err
            )),
        };
    }
    Ok(())
}

/// Stage a first-run key in a dedicated secure slot. The real provider key is
/// not touched until the managed server is healthy and setup is confirmed.
fn stage_pending_inference_config(
    cfg: &InferenceConfig,
    api_key: Option<&str>,
    store: &dyn InferenceCredentialStore,
) -> Result<(), String> {
    stage_pending_inference_config_at(cfg, api_key, store, inference_config_path().as_path())
}

fn stage_pending_inference_config_at(
    cfg: &InferenceConfig,
    api_key: Option<&str>,
    store: &dyn InferenceCredentialStore,
    path: &std::path::Path,
) -> Result<(), String> {
    // A new choice supersedes any interrupted prior attempt. Fail closed if
    // secure storage cannot clear it: otherwise a stale secret could be bound
    // to the wrong engine when boot begins.
    store
        .set(PENDING_INFERENCE_API_KEY, "")
        .map_err(|err| format!("Could not clear a pending API key: {}", err))?;
    if matches!(cfg.kind, SourceKind::Custom) {
        if let Some(api_key) = api_key {
            store
                .set(PENDING_INFERENCE_API_KEY, api_key)
                .map_err(|err| format!("Could not stage the API key: {}", err))?;
        }
    }

    if let Err(config_err) = write_inference_config_at(cfg, path) {
        return match store.set(PENDING_INFERENCE_API_KEY, "") {
            Ok(()) => Err(config_err),
            Err(cleanup_err) => Err(format!(
                "{}; additionally could not discard the staged API key: {}",
                config_err, cleanup_err
            )),
        };
    }
    Ok(())
}

fn discard_pending_inference_setup() -> Result<(), String> {
    let path = inference_config_path();
    if is_offline_distribution() {
        return remove_inference_config_at(&path);
    }
    discard_pending_inference_setup_at(&path, &SystemInferenceCredentialStore)
}

fn discard_pending_inference_setup_at(
    path: &std::path::Path,
    store: &dyn InferenceCredentialStore,
) -> Result<(), String> {
    let key_result = store.set(PENDING_INFERENCE_API_KEY, "");
    let config_result = remove_inference_config_at(path);
    match (key_result, config_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(key_err), Ok(())) => Err(format!("Failed to discard pending API key: {}", key_err)),
        (Ok(()), Err(config_err)) => Err(config_err),
        (Err(key_err), Err(config_err)) => Err(format!(
            "{}; additionally failed to discard pending API key: {}",
            config_err, key_err
        )),
    }
}

/// Rolls a first-run choice back whenever boot exits before confirmation.
/// Confirmed returning-user configs are never removed by a transient outage.
struct PendingInferenceRollback {
    path: Option<std::path::PathBuf>,
    credential: Option<PendingInferenceCredential>,
    credential_store: Option<Arc<dyn InferenceCredentialStore>>,
}

impl PendingInferenceRollback {
    fn new(cfg: &InferenceConfig) -> Self {
        let use_keyring = !is_offline_distribution();
        Self {
            path: (!cfg.confirmed).then_some(inference_config_path()),
            credential: use_keyring
                .then(|| PendingInferenceCredential::for_config(cfg))
                .flatten(),
            credential_store: (use_keyring && !cfg.confirmed).then(|| {
                Arc::new(SystemInferenceCredentialStore) as Arc<dyn InferenceCredentialStore>
            }),
        }
    }

    #[cfg(test)]
    fn at_path(cfg: &InferenceConfig, path: std::path::PathBuf) -> Self {
        Self {
            path: (!cfg.confirmed).then_some(path),
            credential: None,
            credential_store: None,
        }
    }

    #[cfg(test)]
    fn at_path_with_store(
        cfg: &InferenceConfig,
        path: std::path::PathBuf,
        credential_store: Arc<dyn InferenceCredentialStore>,
    ) -> Self {
        Self {
            path: (!cfg.confirmed).then_some(path),
            credential: PendingInferenceCredential::for_config(cfg),
            credential_store: (!cfg.confirmed).then_some(credential_store),
        }
    }

    fn disarm(&mut self) {
        self.path = None;
        self.credential = None;
        self.credential_store = None;
    }

    fn staged_credential_present(&self) -> Result<bool, String> {
        let (Some(_), Some(store)) = (&self.credential, &self.credential_store) else {
            return Ok(false);
        };
        Ok(matches!(
            store.get(PENDING_INFERENCE_API_KEY)?,
            Some(value) if !value.is_empty()
        ))
    }

    /// Commit a staged first-run choice only after every readiness gate has
    /// succeeded.  Keeping this operation on the rollback guard makes all
    /// successful exits use the same write-then-disarm ordering: if the write
    /// fails (or the future is cancelled), `Drop` removes the staged file.
    fn confirm(&mut self, cfg: &mut InferenceConfig) -> Result<(), String> {
        if cfg.confirmed {
            self.disarm();
            return Ok(());
        }

        let path = self
            .path
            .as_ref()
            .ok_or_else(|| "Pending inference setup lost its rollback path.".to_string())?;
        let mut confirmed = cfg.clone();
        confirmed.confirmed = true;

        let mut promoted: Option<(String, Option<String>)> = None;
        if let (Some(credential), Some(store)) = (&self.credential, &self.credential_store) {
            if let Some(staged) = store.get(PENDING_INFERENCE_API_KEY)? {
                if !staged.is_empty() {
                    let previous = store.get(&credential.target_key)?;
                    store.set(&credential.target_key, &staged)?;
                    if let Err(cleanup_err) = store.set(PENDING_INFERENCE_API_KEY, "") {
                        let restore_result = restore_credential(
                            store.as_ref(),
                            &credential.target_key,
                            previous.as_deref(),
                        );
                        return match restore_result {
                            Ok(()) => Err(cleanup_err),
                            Err(restore_err) => Err(format!(
                                "{}; additionally could not restore the previous API key: {}",
                                cleanup_err, restore_err
                            )),
                        };
                    }
                    promoted = Some((credential.target_key.clone(), previous));
                }
            }
        }

        if let Err(config_err) = write_inference_config_at(&confirmed, path) {
            if let (Some((target_key, previous)), Some(store)) = (promoted, &self.credential_store)
            {
                return match restore_credential(store.as_ref(), &target_key, previous.as_deref()) {
                    Ok(()) => Err(config_err),
                    Err(restore_err) => Err(format!(
                        "{}; additionally could not restore the previous API key: {}",
                        config_err, restore_err
                    )),
                };
            }
            return Err(config_err);
        }
        *cfg = confirmed;
        self.disarm();
        Ok(())
    }
}

impl Drop for PendingInferenceRollback {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = std::fs::remove_file(path);
        }
        if self.credential.take().is_some() {
            if let Some(store) = self.credential_store.take() {
                let _ = store.set(PENDING_INFERENCE_API_KEY, "");
            }
        }
    }
}

/// Parse only an explicitly persisted source choice. Missing/invalid config is
/// intentionally `None`: callers on the boot path must wait for setup rather
/// than silently falling back to Ollama.
fn parse_configured_inference_config(text: &str) -> Option<InferenceConfig> {
    let value = serde_json::from_str::<serde_json::Value>(text).ok()?;
    if !value.as_object()?.contains_key("kind") {
        return None;
    }
    let cfg = serde_json::from_value::<InferenceConfig>(value).ok()?;
    if cfg.kind == SourceKind::Cloudflare && !offline_source_allowed(&cfg) {
        return None;
    }
    Some(cfg)
}

fn read_configured_inference_config() -> Option<InferenceConfig> {
    std::fs::read_to_string(inference_config_path())
        .ok()
        .and_then(|text| parse_configured_inference_config(&text))
}

/// Read the on-disk inference config, or the Ollama default if absent.
fn read_inference_config() -> InferenceConfig {
    read_configured_inference_config().unwrap_or_default()
}

fn write_inference_config_at(cfg: &InferenceConfig, path: &std::path::Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let json = serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    std::fs::write(path, json + "\n").map_err(|e| format!("Failed to save inference config: {}", e))
}

/// Upsert `[engine.<engine>] host = "<host>"` into an existing config.toml
/// string, preserving all other content/formatting. Pure: string in, string out.
fn upsert_engine_host(existing: &str, engine: &str, host: &str) -> Result<String, String> {
    let mut doc = existing
        .parse::<toml_edit::DocumentMut>()
        .map_err(|e| format!("Invalid config.toml: {}", e))?;
    doc["engine"][engine]["host"] = toml_edit::value(host);
    Ok(doc.to_string())
}

/// Write the custom-endpoint host into ~/.openjarvis/config.toml so
/// `jarvis serve` (which reads that file via load_config) points at it.
/// The `<ENGINE>_HOST` env var is unreliable — it is shadowed by the engine's
/// non-empty default host in the Python layer — so config.toml is the override.
fn set_engine_host_in_config(engine: &str, host: &str) -> Result<(), String> {
    let path = std::path::PathBuf::from(home_dir())
        .join(".openjarvis")
        .join("config.toml");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let existing = match std::fs::read_to_string(&path) {
        Ok(config) => config,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => return Err(format!("Failed to read config.toml: {error}")),
    };
    let updated = upsert_engine_host(&existing, engine, host)?;
    std::fs::write(&path, updated).map_err(|e| format!("Failed to write config.toml: {}", e))
}

/// Normalize a user-entered server URL to a bare base host: trim whitespace,
/// drop a trailing `/v1` segment (the engine re-appends its own api prefix),
/// then drop any trailing slash.
fn normalize_host(raw: &str) -> String {
    let s = raw.trim().trim_end_matches('/');
    let s = s.strip_suffix("/v1").unwrap_or(s);
    s.trim_end_matches('/').to_string()
}

/// Check speech backend health.
#[tauri::command]
async fn speech_health(api_url: String) -> Result<serde_json::Value, String> {
    let url = format!("{}/v1/speech/health", api_url);
    let resp = reqwest::get(&url)
        .await
        .map_err(|e| format!("Connection failed: {}", e))?;
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Invalid response: {}", e))?;
    Ok(body)
}

// ---------------------------------------------------------------------------
// Native macOS overlay — NSPanel + WKWebView, entirely bypassing Tauri's
// window management so we get proper always-on-top, transparency, non-
// activating panel behaviour and cross-Space support.
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod native_overlay {
    use objc::declare::ClassDecl;
    use objc::runtime::{Class, Object, Sel, BOOL, NO, YES};
    use objc::{class, msg_send, sel, sel_impl};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Raw pointer to the NSPanel, stored as usize for atomicity.
    static PANEL_PTR: AtomicUsize = AtomicUsize::new(0);
    /// Raw pointer to the WKWebView inside the panel.
    static WEBVIEW_PTR: AtomicUsize = AtomicUsize::new(0);
    /// Raw pointer to the previously-frontmost NSRunningApplication.
    static PREV_APP: AtomicUsize = AtomicUsize::new(0);

    // CoreGraphics geometry types expected by AppKit.
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct CGPoint {
        x: f64,
        y: f64,
    }
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct CGSize {
        width: f64,
        height: f64,
    }
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct CGRect {
        origin: CGPoint,
        size: CGSize,
    }

    /// Create an autoreleased NSString from a Rust &str.
    unsafe fn nsstring(s: &str) -> *mut Object {
        let obj: *mut Object = msg_send![class!(NSString), alloc];
        msg_send![obj,
            initWithBytes: s.as_ptr()
            length: s.len()
            encoding: 4usize  // NSUTF8StringEncoding
        ]
    }

    // ------------------------------------------------------------------
    // Conversation persistence
    // ------------------------------------------------------------------

    fn conversation_path() -> std::path::PathBuf {
        std::path::PathBuf::from(super::home_dir())
            .join(".openjarvis")
            .join("overlay-conversation.json")
    }

    pub fn load_conversation() -> String {
        std::fs::read_to_string(conversation_path()).unwrap_or_else(|_| "[]".into())
    }

    /// Read cloud API keys and return a JSON array of model IDs
    /// whose provider has a key configured.
    fn cloud_models_json() -> String {
        let keys = super::read_cloud_keys();
        let mut models: Vec<&str> = Vec::new();
        for (name, value) in &keys {
            if value.is_empty() {
                continue;
            }
            match name.as_str() {
                "OPENAI_API_KEY" => models.extend(["gpt-4o", "gpt-4o-mini"]),
                "ANTHROPIC_API_KEY" => {
                    models.extend(["claude-sonnet-4-20250514", "claude-haiku-4-20250414"])
                }
                "GEMINI_API_KEY" | "GOOGLE_API_KEY" => {
                    models.extend(["gemini-2.5-flash", "gemini-2.5-pro"])
                }
                _ => {}
            }
        }
        serde_json::to_string(&models).unwrap_or_else(|_| "[]".into())
    }

    fn save_conversation(json: &str) {
        let path = conversation_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, json);
    }

    /// Apply every transparency trick to the WKWebView.
    /// Called once at creation and again after the page finishes loading.
    unsafe fn force_transparent(wv: *mut Object) {
        let clear: *mut Object = msg_send![class!(NSColor), clearColor];
        let _: () = msg_send![wv, _setDrawsBackground: NO];
        let no_num: *mut Object = msg_send![class!(NSNumber), numberWithBool: NO];
        let _: () = msg_send![wv, setValue: no_num forKey: nsstring("drawsBackground")];
        let _: () = msg_send![wv, setUnderPageBackgroundColor: clear];
        // Also inject CSS to nuke any remaining background
        let js = nsstring(
            "document.documentElement.style.background='transparent';\
             document.body.style.background='transparent';",
        );
        let nil: *mut Object = std::ptr::null_mut();
        let _: () = msg_send![wv, evaluateJavaScript: js completionHandler: nil];
    }

    // ------------------------------------------------------------------
    // Public API (must be called on the main thread)
    // ------------------------------------------------------------------

    /// Build the native overlay panel.  Call once during app setup.
    pub unsafe fn create(html: &str, api_port: u16) {
        // --- Custom NSPanel subclass that accepts keyboard input ------
        if Class::get("JarvisOverlayPanel").is_none() {
            let sup = Class::get("NSPanel").unwrap();
            let mut decl = ClassDecl::new("JarvisOverlayPanel", sup).unwrap();
            extern "C" fn yes(_: &Object, _: Sel) -> BOOL {
                YES
            }
            decl.add_method(
                sel!(canBecomeKeyWindow),
                yes as extern "C" fn(&Object, Sel) -> BOOL,
            );
            decl.register();
        }

        // --- WKNavigationDelegate — re-apply transparency after load --
        if Class::get("JarvisOverlayNavDelegate").is_none() {
            let sup = Class::get("NSObject").unwrap();
            let mut decl = ClassDecl::new("JarvisOverlayNavDelegate", sup).unwrap();
            extern "C" fn did_finish(_: &Object, _: Sel, wv: *mut Object, _nav: *mut Object) {
                unsafe {
                    force_transparent(wv);
                }
            }
            decl.add_method(
                sel!(webView:didFinishNavigation:),
                did_finish as extern "C" fn(&Object, Sel, *mut Object, *mut Object),
            );
            decl.register();
        }

        // --- WKScriptMessageHandler so JS can call hide() ------------
        if Class::get("JarvisOverlayMsgHandler").is_none() {
            let sup = Class::get("NSObject").unwrap();
            let mut decl = ClassDecl::new("JarvisOverlayMsgHandler", sup).unwrap();
            extern "C" fn on_msg(_: &Object, _: Sel, _ctrl: *mut Object, msg: *mut Object) {
                unsafe {
                    let body: *mut Object = msg_send![msg, body];
                    if body.is_null() {
                        return;
                    }
                    let c: *const std::os::raw::c_char = msg_send![body, UTF8String];
                    if c.is_null() {
                        return;
                    }
                    if let Ok(s) = std::ffi::CStr::from_ptr(c).to_str() {
                        if s == "hide" {
                            hide();
                        } else if let Some(json) = s.strip_prefix("save:") {
                            save_conversation(json);
                        } else if let Some(coords) = s.strip_prefix("drag:") {
                            drag(coords);
                        }
                    }
                }
            }
            decl.add_method(
                sel!(userContentController:didReceiveScriptMessage:),
                on_msg as extern "C" fn(&Object, Sel, *mut Object, *mut Object),
            );
            decl.register();
        }

        // --- Create the NSPanel --------------------------------------
        let frame = CGRect {
            origin: CGPoint { x: 0.0, y: 0.0 },
            size: CGSize {
                width: 560.0,
                height: 400.0,
            },
        };
        // NSWindowStyleMaskNonactivatingPanel = 1 << 7
        let style: u64 = 1 << 7;

        let cls = Class::get("JarvisOverlayPanel").unwrap();
        let panel: *mut Object = msg_send![cls, alloc];
        let panel: *mut Object = msg_send![panel,
            initWithContentRect: frame
            styleMask: style
            backing: 2u64       // NSBackingStoreBuffered
            defer: NO
        ];

        // Window level — NSFloatingWindowLevel (3).
        let _: () = msg_send![panel, setLevel: 3_i64];
        // canJoinAllSpaces (1) | fullScreenAuxiliary (1<<8)
        let _: () = msg_send![panel, setCollectionBehavior: 257_u64];
        let _: () = msg_send![panel, setHidesOnDeactivate: NO];
        let _: () = msg_send![panel, setOpaque: NO];
        let _: () = msg_send![panel, setHasShadow: NO];
        let _: () = msg_send![panel, setMovableByWindowBackground: YES];

        let clear: *mut Object = msg_send![class!(NSColor), clearColor];
        let _: () = msg_send![panel, setBackgroundColor: clear];
        let _: () = msg_send![panel, center];

        // --- WKWebView -----------------------------------------------
        let cfg: *mut Object = msg_send![class!(WKWebViewConfiguration), alloc];
        let cfg: *mut Object = msg_send![cfg, init];

        // Attach message handler ("overlay" channel)
        let hcls = Class::get("JarvisOverlayMsgHandler").unwrap();
        let handler: *mut Object = msg_send![hcls, alloc];
        let handler: *mut Object = msg_send![handler, init];
        let uc: *mut Object = msg_send![cfg, userContentController];
        let _: () = msg_send![uc,
            addScriptMessageHandler: handler
            name: nsstring("overlay")
        ];

        let wv: *mut Object = msg_send![class!(WKWebView), alloc];
        let wv: *mut Object = msg_send![wv,
            initWithFrame: frame
            configuration: cfg
        ];

        // ---- Make the webview fully transparent ----
        force_transparent(wv);

        // Set navigation delegate so we re-apply after page loads
        let nav_cls = Class::get("JarvisOverlayNavDelegate").unwrap();
        let nav_del: *mut Object = msg_send![nav_cls, alloc];
        let nav_del: *mut Object = msg_send![nav_del, init];
        let _: () = msg_send![wv, setNavigationDelegate: nav_del];

        let _: () = msg_send![panel, setContentView: wv];
        WEBVIEW_PTR.store(wv as usize, Ordering::SeqCst);

        // Inject saved conversation into the HTML template, then load it.
        // Use the API server as the base URL so fetch() is same-origin.
        // Escape "</" so the JSON can't prematurely close the <script> tag.
        // ("\/" is valid JSON — resolves back to "/" when parsed.)
        let saved = load_conversation().replace("</", "<\\/");
        let cloud = cloud_models_json();
        let filled = html
            .replace("__SAVED_MESSAGES__", &saved)
            .replace("__CLOUD_MODELS__", &cloud);
        let base_str = nsstring(&format!("http://127.0.0.1:{}", api_port));
        let base_url: *mut Object = msg_send![class!(NSURL), URLWithString: base_str];
        let _: () = msg_send![wv,
            loadHTMLString: nsstring(&filled)
            baseURL: base_url
        ];

        PANEL_PTR.store(panel as usize, Ordering::SeqCst);
    }

    pub unsafe fn toggle() {
        let ptr = PANEL_PTR.load(Ordering::SeqCst);
        if ptr == 0 {
            return;
        }
        let panel = ptr as *mut Object;
        let vis: BOOL = msg_send![panel, isVisible];
        if vis != NO {
            hide();
        } else {
            show();
        }
    }

    pub unsafe fn show() {
        let ptr = PANEL_PTR.load(Ordering::SeqCst);
        if ptr == 0 {
            return;
        }
        let panel = ptr as *mut Object;

        // Re-apply transparency every time (the webview can reset it)
        let wv_ptr = WEBVIEW_PTR.load(Ordering::SeqCst);
        if wv_ptr != 0 {
            force_transparent(wv_ptr as *mut Object);
        }

        // Remember the currently-frontmost app so we can restore it.
        let ws: *mut Object = msg_send![class!(NSWorkspace), sharedWorkspace];
        let front: *mut Object = msg_send![ws, frontmostApplication];
        if !front.is_null() {
            let _: () = msg_send![front, retain];
            let old = PREV_APP.swap(front as usize, Ordering::SeqCst);
            if old != 0 {
                let _: () = msg_send![(old as *mut Object), release];
            }
        }

        // Activate our process so the panel receives keyboard input.
        let app: *mut Object = msg_send![class!(NSApplication), sharedApplication];
        let _: () = msg_send![app, activateIgnoringOtherApps: YES];
        let nil: *mut Object = std::ptr::null_mut();
        let _: () = msg_send![panel, makeKeyAndOrderFront: nil];

        // Focus the text field inside the webview.
        let wv: *mut Object = msg_send![panel, contentView];
        let js = nsstring("document.getElementById('input').focus()");
        let _: () = msg_send![wv, evaluateJavaScript: js completionHandler: nil];
    }

    /// Move the panel by a screen-space delta (called from JS drag handler).
    unsafe fn drag(coords: &str) {
        let ptr = PANEL_PTR.load(Ordering::SeqCst);
        if ptr == 0 {
            return;
        }
        let panel = ptr as *mut Object;
        let Some((dxs, dys)) = coords.split_once(',') else {
            return;
        };
        let Ok(dx) = dxs.parse::<f64>() else { return };
        let Ok(dy) = dys.parse::<f64>() else { return };
        // NSWindow frame origin is bottom-left; screen Y increases upward,
        // but mouse screenY increases downward, so invert dy.
        let frame: CGRect = msg_send![panel, frame];
        let origin = CGPoint {
            x: frame.origin.x + dx,
            y: frame.origin.y - dy,
        };
        let _: () = msg_send![panel, setFrameOrigin: origin];
    }

    pub unsafe fn hide() {
        let ptr = PANEL_PTR.load(Ordering::SeqCst);
        if ptr == 0 {
            return;
        }
        let panel = ptr as *mut Object;
        let nil: *mut Object = std::ptr::null_mut();
        let _: () = msg_send![panel, orderOut: nil];

        // Give focus back to whatever app was frontmost before.
        let prev = PREV_APP.swap(0, Ordering::SeqCst);
        if prev != 0 {
            let prev_app = prev as *mut Object;
            let _: BOOL = msg_send![prev_app, activateWithOptions: 2_u64];
            let _: () = msg_send![prev_app, release];
        }
    }
}

/// Dispatch a closure onto the main thread via GCD.
#[cfg(target_os = "macos")]
fn on_main_thread(f: impl FnOnce() + Send + 'static) {
    dispatch::Queue::main().exec_async(f);
}

// ---------------------------------------------------------------------------
// Overlay Tauri commands (thin wrappers that dispatch to the main thread)
// ---------------------------------------------------------------------------

#[tauri::command]
async fn get_overlay_conversation() -> Result<String, String> {
    #[cfg(target_os = "macos")]
    {
        return Ok(native_overlay::load_conversation());
    }
    #[cfg(not(target_os = "macos"))]
    Ok("[]".into())
}

#[tauri::command]
async fn toggle_overlay() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    on_main_thread(|| unsafe { native_overlay::toggle() });
    Ok(())
}

#[tauri::command]
async fn hide_overlay() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    on_main_thread(|| unsafe { native_overlay::hide() });
    Ok(())
}

// ---------------------------------------------------------------------------
// App entry point
// ---------------------------------------------------------------------------

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[cfg(target_os = "windows")]
    let _ = capture_cloudflare_launch_environment();
    let backend: SharedBackend = Arc::new(Mutex::new(BackendManager::default()));
    let configured_at_launch = match read_configured_inference_config() {
        Some(cfg) if cfg.confirmed => {
            // A confirmed source never consumes a staging slot. Clear any
            // remnant from an interrupted standard-edition config write.
            if !is_offline_distribution() {
                let _ = secure_store_set(PENDING_INFERENCE_API_KEY, "");
            }
            Some(cfg)
        }
        Some(_) => {
            // A prior first-run boot was interrupted or failed before it
            // reached ready. Never replay that unvalidated choice on launch.
            let _ = discard_pending_inference_setup();
            None
        }
        None => {
            // Clean up a stage left behind if the process exited between the
            // secure-store write and the pending config write.
            if !is_offline_distribution() {
                let _ = secure_store_set(PENDING_INFERENCE_API_KEY, "");
            }
            None
        }
    };
    let initial_status = configured_at_launch
        .as_ref()
        .map(SetupStatus::starting)
        .unwrap_or_default();
    let status: SharedStatus = Arc::new(Mutex::new(initial_status));

    let boot_backend_ref = backend.clone();
    let boot_status_ref = status.clone();

    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .manage(backend.clone())
        .manage(status.clone())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_autostart::init(
            MacosLauncher::LaunchAgent,
            Some(vec!["--hidden"]),
        ))
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(move |app| {
            // System tray
            let show = MenuItemBuilder::with_id("show", "Abrir A.C.E.").build(app)?;
            let health = MenuItemBuilder::with_id("health", "Health: starting...")
                .enabled(false)
                .build(app)?;
            let quit = MenuItemBuilder::with_id("quit", "Sair do A.C.E.").build(app)?;

            let menu = MenuBuilder::new(app)
                .item(&show)
                .separator()
                .item(&health)
                .separator()
                .item(&quit)
                .build()?;

            let _tray = TrayIconBuilder::with_id("main")
                .icon(app.default_window_icon().unwrap().clone())
                .tooltip("A.C.E.")
                .menu(&menu)
                .on_menu_event(move |app, event| match event.id().as_ref() {
                    "show" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.unminimize();
                            let _ = window.set_focus();
                        }
                    }
                    "quit" => {
                        app.exit(0);
                    }
                    _ => {}
                })
                .build(app)?;

            // Create native macOS overlay panel
            #[cfg(target_os = "macos")]
            unsafe {
                native_overlay::create(include_str!("overlay.html"), JARVIS_PORT);
            }

            // Register Cmd+Shift+Space to toggle the overlay
            {
                #[cfg(target_os = "macos")]
                use tauri_plugin_global_shortcut::{
                    Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState,
                };
                #[cfg(target_os = "macos")]
                let sc = Shortcut::new(Some(Modifiers::META | Modifiers::SHIFT), Code::Space);
                #[cfg(target_os = "macos")]
                if let Err(e) = app.global_shortcut().on_shortcut(sc, |_app, _sc, ev| {
                    if ev.state == ShortcutState::Pressed {
                        #[cfg(target_os = "macos")]
                        unsafe {
                            native_overlay::toggle();
                        }
                    }
                }) {
                    eprintln!("Warning: could not register Cmd+Shift+Space: {e}");
                }
            }

            // Returning users keep automatic startup after they have persisted
            // a source. Fresh installs stay inert until the setup UI records
            // explicit consent and invokes `start_backend`.
            if is_offline_distribution() {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.set_title("A.C.E.");
                }
                let root = app.path().resolve("ace-backend", tauri::path::BaseDirectory::Resource).ok();
                let backend = boot_backend_ref.clone();
                let status = boot_status_ref.clone();
                tauri::async_runtime::spawn(async move {
                    backend.lock().await.bundled_root = root;
                    start_managed_boot(backend, status, true).await;
                });
            } else if configured_at_launch.is_some() {
                tauri::async_runtime::spawn(start_managed_boot(boot_backend_ref, boot_status_ref, true));
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_setup_status,
            get_api_base,
            is_offline_distribution,
            start_backend,
            stop_backend,
            reset_inference_source,
            check_health,
            check_packaged_backend_health,
            fetch_energy,
            fetch_telemetry,
            fetch_traces,
            fetch_trace,
            fetch_learning_stats,
            fetch_learning_policy,
            fetch_memory_stats,
            search_memory,
            fetch_agents,
            fetch_models,
            run_jarvis_command,
            fetch_savings,
            submit_savings,
            transcribe_audio,
            speech_health,
            list_ollama_models,
            pull_ollama_model,
            delete_ollama_model,
            save_cloud_key,
            get_cloud_key_status,
            cloudflare_credentials::get_cloudflare_credentials_status,
            cloudflare_credentials::save_cloudflare_credentials,
            cloudflare_credentials::remove_cloudflare_credentials,
            get_inference_source,
            set_inference_source,
            toggle_overlay,
            hide_overlay,
            get_overlay_conversation,
        ])
        .build(tauri::generate_context!())
        .expect("error while building OpenJarvis Desktop")
        .run(move |_app, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                let b = backend.clone();
                tauri::async_runtime::spawn(async move {
                    b.lock().await.stop_all().await;
                });
            }
        });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        boot_plan, discard_pending_inference_setup_at, format_extension_import_failure,
        format_missing_rust_toolchain, format_port_unavailable, format_uv_sync_failure,
        format_uv_sync_spawn_error, managed_cloud_key_names, matching_installed_model,
        model_names_match, normalize_host, normalize_ollama_model_tag,
        parse_configured_inference_config, parse_ollama_model_names, parse_ollama_models,
        persist_confirmed_inference_config_at, project_root_near_executable,
        reload_cloud_keys_for_owned_backend_at, spawn_owned_child,
        stage_pending_inference_config_at, upsert_engine_host, uv_sync_stderr_tail, BackendManager,
        InferenceConfig, InferenceCredentialStore, PendingInferenceRollback, SetupStatus,
        SourceKind, DESKTOP_UV_SYNC_COMMAND, PENDING_INFERENCE_API_KEY,
    };
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Mutex as StdMutex;

    #[test]
    fn atlas_cloud_key_is_loaded_with_managed_desktop_keys() {
        assert!(managed_cloud_key_names().contains(&"ATLASCLOUD_API_KEY".to_string()));
    }

    #[test]
    fn offline_ollama_arguments_pin_the_installed_model_and_loopback_host() {
        let cfg = InferenceConfig {
            confirmed: true,
            model: Some("test-model".into()),
            ..InferenceConfig::default()
        };
        let plan = boot_plan(&cfg).expect("valid Ollama plan");
        let args = super::offline_serve_args(&plan, Some("test-model:latest"));
        assert!(args.windows(2).any(|args| args == ["--model", "test-model:latest"]));
        assert!(args.windows(2).any(|args| args == ["--local-engine-host", "http://127.0.0.1:11434"]));
    }

    #[test]
    fn offline_custom_arguments_preserve_the_selected_local_endpoint() {
        let cfg = InferenceConfig {
            kind: SourceKind::Custom,
            confirmed: true,
            model: Some("test-model".into()),
            host: Some("http://[::1]:1234".into()),
            engine: Some("lmstudio".into()),
        };
        assert!(super::offline_source_allowed(&cfg));
        let plan = boot_plan(&cfg).expect("valid custom plan");
        let args = super::offline_serve_args(&plan, None);
        assert!(args.windows(2).any(|args| args == ["--local-engine-host", "http://[::1]:1234"]));
    }

    #[test]
    fn cloudflare_source_is_explicit_and_validates_model_ids() {
        use super::CLOUDFLARE_MODEL;

        let cfg = parse_configured_inference_config(r#"{"kind":"cloudflare"}"#).unwrap();
        assert!(cfg.confirmed);
        assert!(super::offline_source_allowed(&cfg));
        assert_eq!(SetupStatus::starting(&cfg).source, "cloudflare");
        let plan = boot_plan(&cfg).unwrap();
        assert!(!plan.launch_ollama);
        assert!(plan.engine_host.is_none());
        assert!(plan.model_to_pull.is_none());
        assert_eq!(
            super::offline_serve_args(&plan, None),
            ["--engine", "cloudflare", "--model", CLOUDFLARE_MODEL, "--agent", "simple"],
        );
        for (host, engine, api_key) in [(true, false, false), (false, true, false), (false, false, true)] {
            assert!(super::validate_cloudflare_source(Some(CLOUDFLARE_MODEL), host, engine, api_key).is_err());
        }
        assert!(super::validate_cloudflare_source(Some(""), false, false, false).is_err());
        for id in ["@cf/meta/example-8b", "@hf/vendor/model.v2"] {
            let cfg = InferenceConfig { model: Some(format!("  {id}  ")), ..cfg.clone() };
            assert_eq!(boot_plan(&cfg).unwrap().serve_args[3], id);
        }
        for invalid in [
            "", ".", "@cf/meta/", "@cf/meta/../model", "@cf/meta/.hidden", "@hf/-vendor/model",
            "@cf/vendor/model/path", "@cf/vendor/model?option=true", "@other/vendor/model",
            "@cf/vendor/módelo", "https://example.org/model",
        ] {
            let cfg = InferenceConfig { model: Some(invalid.into()), ..cfg.clone() };
            assert!(boot_plan(&cfg).is_err(), "accepted {invalid}");
        }
        let too_long = format!("@cf/vendor/{}", "a".repeat(120));
        assert!(super::validate_cloudflare_source(Some(&too_long), false, false, false).is_err());
        assert!(parse_configured_inference_config(r#"{"kind":"cloudflare","model":"different"}"#).is_none());
        assert!(parse_configured_inference_config(r#"{"kind":"cloudflare","host":"http://127.0.0.1"}"#).is_none());
        assert!(boot_plan(&InferenceConfig {
            kind: SourceKind::Custom,
            engine: Some(" CloudFlare ".into()),
            ..cfg
        }).is_err());
    }

    #[test]
    fn packaged_commands_only_pass_cloudflare_variables_to_the_selected_worker() {
        use std::ffi::OsStr;
        use super::{configure_bundled_server, local_command, offline::AttachedBackendMode, BundledSource, CLOUDFLARE_ENV, CLOUDFLARE_MODEL};

        let credentials = [
            (CLOUDFLARE_ENV[0], "synthetic-account".into()),
            (CLOUDFLARE_ENV[1], "synthetic-token".into()),
        ];
        let cloudflare = boot_plan(&InferenceConfig {
            kind: SourceKind::Cloudflare,
            ..InferenceConfig::default()
        }).unwrap();

        let mut local = local_command("test-executable");
        assert_eq!(
            configure_bundled_server(&mut local, 4738, BundledSource::LocalServices),
            AttachedBackendMode::LocalServices,
        );
        for name in CLOUDFLARE_ENV {
            assert!(local.as_std().get_envs().any(|(key, value)| key == name && value.is_none()));
        }
        assert_eq!(local.as_std().get_args().last(), Some(OsStr::new("--local-services-only")));

        let mut cloud = local_command("test-executable");
        assert_eq!(
            configure_bundled_server(&mut cloud, 4738, BundledSource::Cloudflare {
                args: &cloudflare.serve_args,
                credentials: &credentials,
            }),
            AttachedBackendMode::Cloudflare,
        );
        assert_eq!(
            cloud.as_std().get_args().collect::<Vec<_>>(),
            ["serve", "--host", "127.0.0.1", "--port", "4738", "--engine", "cloudflare",
             "--model", CLOUDFLARE_MODEL, "--agent", "simple"],
        );
        for (name, expected) in &credentials {
            assert!(cloud.as_std().get_envs().any(|(key, value)| {
                key == *name && value == Some(OsStr::new(expected))
            }));
        }
        for name in super::MANAGED_CLOUD_KEY_NAMES {
            assert!(cloud.as_std().get_envs().any(|(key, value)| key == *name && value.is_none()));
        }
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_launch_environment_is_scrubbed_before_starting_helpers() {
        use super::{capture_cloudflare_launch_environment, CLOUDFLARE_ENV};

        let child_marker = "ACE_TEST_CLOUDFLARE_LAUNCH_ENV_CHILD";
        if std::env::var_os(child_marker).is_some() {
            let captured = capture_cloudflare_launch_environment().as_ref().unwrap();
            assert_eq!(captured[0].1, "fixture-account");
            assert_eq!(captured[1].1, "fixture-token");
            assert!(CLOUDFLARE_ENV.into_iter().all(|name| std::env::var_os(name).is_none()));
            return;
        }

        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::windows_launch_environment_is_scrubbed_before_starting_helpers"])
            .env(child_marker, "1")
            .env(CLOUDFLARE_ENV[0], "fixture-account")
            .env(CLOUDFLARE_ENV[1], "fixture-token")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn packaged_cloudflare_attachment_requires_a_confirmed_matching_choice() {
        use super::offline::AttachedBackendMode::*;
        use super::may_attach_bundled_mode;

        let cloud = InferenceConfig {
            kind: SourceKind::Cloudflare,
            confirmed: true,
            ..InferenceConfig::default()
        };
        let local = InferenceConfig {
            confirmed: true,
            ..InferenceConfig::default()
        };
        let pending = InferenceConfig { confirmed: false, ..cloud.clone() };
        let gemma = Some(super::CLOUDFLARE_MODEL);
        assert!(may_attach_bundled_mode(Cloudflare, gemma, Some(&cloud), true));
        assert!(!may_attach_bundled_mode(Cloudflare, Some("@cf/vendor/alternative"), Some(&cloud), true));
        assert!(!may_attach_bundled_mode(Cloudflare, None, Some(&cloud), true));
        assert!(!may_attach_bundled_mode(Cloudflare, gemma, Some(&pending), true));
        assert!(!may_attach_bundled_mode(Cloudflare, gemma, Some(&cloud), false));
        assert!(!may_attach_bundled_mode(Cloudflare, gemma, Some(&local), true));
        assert!(!may_attach_bundled_mode(Cloudflare, gemma, None, true));
        assert!(!may_attach_bundled_mode(Inference, None, Some(&cloud), true));
        assert!(may_attach_bundled_mode(Inference, None, Some(&local), true));
        assert!(may_attach_bundled_mode(LocalServices, None, Some(&cloud), false));
    }

    #[cfg(feature = "windows-offline")]
    #[tokio::test]
    #[ignore = "requires a staged Windows backend and fresh OPENJARVIS_HOME"]
    async fn packaged_desktop_reattaches_cloudflare_only_after_matching_selection() {
        use super::{
            configure_bundled_server, offline::{AttachedBackendMode, BundledBackend},
            BundledSource, ChildHandle, JarvisHealthTarget, JarvisStartResult, CLOUDFLARE_ENV,
            CLOUDFLARE_MODEL,
        };

        let runtime = std::env::var_os("ACE_TEST_BUNDLED_BACKEND")
            .map(std::path::PathBuf::from)
            .expect("Set ACE_TEST_BUNDLED_BACKEND to the packaged backend directory.");
        let home = std::env::var_os("OPENJARVIS_HOME")
            .map(std::path::PathBuf::from)
            .expect("Set OPENJARVIS_HOME to a new test directory.");
        assert!(home.is_absolute() && !home.exists());
        assert!(std::env::var_os("OPENJARVIS_CONFIG").is_none());
        let bundled = BundledBackend::open(&runtime).unwrap();
        bundled.initialize_profile().await.unwrap();

        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let cloud = InferenceConfig {
            kind: SourceKind::Cloudflare,
            confirmed: true,
            model: Some(CLOUDFLARE_MODEL.into()),
            ..InferenceConfig::default()
        };
        let plan = boot_plan(&cloud).unwrap();
        // Synthetic values only. Startup and health perform no Cloudflare generation.
        let credentials = [(CLOUDFLARE_ENV[0], "f".repeat(32)), (CLOUDFLARE_ENV[1], "z".repeat(40))];
        let mut command = bundled.command();
        let mode = configure_bundled_server(&mut command, port, BundledSource::Cloudflare {
            args: &plan.serve_args,
            credentials: &credentials,
        });
        command.stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
        let child = super::spawn_owned_child(&mut command).unwrap();
        let pid = child.id().unwrap();
        let backend = super::SharedBackend::new(tokio::sync::Mutex::new(BackendManager {
            jarvis: Some(ChildHandle { child }),
            bundled_root: Some(runtime),
            ..BackendManager::default()
        }));

        let result = super::wait_for_jarvis_health(
            JarvisHealthTarget::Packaged { runtime: &bundled, port, mode },
            std::time::Duration::from_secs(25),
            &backend,
        ).await;
        assert!(matches!(result, JarvisStartResult::Ready), "{result:?}");
        let mut child = backend.lock().await.jarvis.take().unwrap().child;
        let status = super::SharedStatus::new(tokio::sync::Mutex::new(SetupStatus::default()));
        assert!(super::try_attach_bundled_server(&bundled, port, &backend, &status, true).await);
        assert!(!status.lock().await.server_ready);
        assert!(status.lock().await.error.as_ref().unwrap().contains("Return to local services"));
        let identity = backend.lock().await.attached.clone().unwrap();
        assert_eq!(identity.pid, pid);
        assert_eq!(identity.mode, AttachedBackendMode::Cloudflare);
        assert_eq!(identity.cloudflare_model.as_deref(), Some(CLOUDFLARE_MODEL));
        backend.lock().await.stop_children().await;
        assert!(child.try_wait().unwrap().is_none(), "Unselected cloud worker was stopped.");

        let different_cloud = InferenceConfig {
            model: Some("@cf/vendor/different-model".into()),
            ..cloud.clone()
        };
        super::write_inference_config_at(&different_cloud, &home.join("inference-offline.json")).unwrap();
        assert!(super::try_attach_bundled_server(&bundled, port, &backend, &status, true).await);
        assert!(!status.lock().await.server_ready);

        super::write_inference_config_at(&cloud, &home.join("inference-offline.json")).unwrap();
        assert!(super::try_attach_bundled_server(&bundled, port, &backend, &status, true).await);
        assert!(status.lock().await.server_ready);
        assert_eq!(status.lock().await.source, "cloudflare");
        assert_eq!(backend.lock().await.attached.as_ref(), Some(&identity));
        super::stop_attached_backend(&backend, port).await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(10), child.wait())
            .await
            .expect("Explicit recovery must stop the verified backend.")
            .unwrap();
    }

    #[test]
    fn packaged_health_distinguishes_available_storage_from_inference() {
        use super::PackagedBackendHealth::*;
        use reqwest::StatusCode;
        use serde_json::json;

        for (status, body, expected) in [
            (StatusCode::OK, json!({"status": "ok"}), Inference),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                json!({"detail": "Engine unhealthy"}),
                InferenceUnavailable,
            ),
            (
                StatusCode::OK,
                json!({
                    "status": "ok",
                    "mode": "local_services_only",
                    "inference_available": false,
                }),
                LocalServices,
            ),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                json!({"detail": "Unknown service"}),
                Unreachable,
            ),
            (
                StatusCode::OK,
                json!({"status": "ok", "mode": "unknown"}),
                Unreachable,
            ),
            (
                StatusCode::TEMPORARY_REDIRECT,
                json!({"status": "ok"}),
                Unreachable,
            ),
        ] {
            assert_eq!(super::classify_packaged_health(status, &body), expected);
        }
    }

    #[cfg(feature = "windows-offline")]
    #[tokio::test]
    #[ignore = "requires a staged Windows backend, fresh OPENJARVIS_HOME, and free port 8000"]
    async fn packaged_backend_boots_without_an_inference_source() {
        let runtime = std::env::var_os("ACE_TEST_BUNDLED_BACKEND")
            .map(std::path::PathBuf::from)
            .expect("Set ACE_TEST_BUNDLED_BACKEND to the packaged backend directory.");
        let home = std::env::var_os("OPENJARVIS_HOME")
            .map(std::path::PathBuf::from)
            .expect("Set OPENJARVIS_HOME to a new test directory.");
        assert!(home.is_absolute() && !home.exists());
        assert!(std::env::var_os("OPENJARVIS_CONFIG").is_none());
        let backend = super::SharedBackend::new(tokio::sync::Mutex::new(BackendManager {
            bundled_root: Some(runtime),
            ..BackendManager::default()
        }));
        let status = super::SharedStatus::new(tokio::sync::Mutex::new(SetupStatus::default()));
        super::boot_backend(backend.clone(), status.clone(), true).await;
        let state = status.lock().await.clone();
        assert_eq!(state.phase, "local_services_ready", "{:?}", state.error);
        assert!(state.server_ready);
        assert!(!state.model_ready);
        assert!(home.join("config.toml").is_file());
        assert!(!home.join("inference-offline.json").exists());
        backend.lock().await.stop_children().await;
    }

    #[test]
    fn packaged_api_requires_one_exact_revision() {
        use reqwest::header::{HeaderMap, HeaderValue};

        for (values, compatible) in [
            (&[][..], false),
            (&["0"][..], false),
            (&["999"][..], false),
            (&["01"][..], false),
            (&["1, 1"][..], false),
            (&[super::PACKAGED_BACKEND_API_REVISION, super::PACKAGED_BACKEND_API_REVISION][..], false),
            (&[super::PACKAGED_BACKEND_API_REVISION][..], true),
        ] {
            let mut headers = HeaderMap::new();
            for &value in values {
                headers.append(super::PACKAGED_BACKEND_API_HEADER, HeaderValue::from_static(value));
            }
            assert_eq!(super::matches_packaged_api(&headers), compatible, "{values:?}");
        }
    }

    #[cfg(feature = "windows-offline")]
    #[tokio::test]
    #[ignore = "requires a staged Windows backend and fresh OPENJARVIS_HOME"]
    async fn packaged_desktop_reattaches_without_starting_or_stopping_a_second_server() {
        exercise_packaged_reattachment("ACE_TEST_BUNDLED_BACKEND", true).await;
    }

    #[cfg(feature = "windows-offline")]
    #[tokio::test]
    #[ignore = "requires a previous Windows backend without an API revision and fresh OPENJARVIS_HOME"]
    async fn packaged_desktop_keeps_legacy_backend_for_explicit_recovery() {
        exercise_packaged_reattachment("ACE_TEST_LEGACY_BACKEND", false).await;
    }

    #[cfg(feature = "windows-offline")]
    async fn exercise_packaged_reattachment(runtime_variable: &str, compatible: bool) {
        use super::{offline, ChildHandle, JarvisStartResult};
        let runtime = std::env::var_os(runtime_variable)
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| panic!("Set {runtime_variable} to the packaged backend directory."));
        let home = std::env::var_os("OPENJARVIS_HOME")
            .map(std::path::PathBuf::from)
            .expect("Set OPENJARVIS_HOME to a new test directory.");
        assert!(home.is_absolute() && !home.exists());
        assert!(std::env::var_os("OPENJARVIS_CONFIG").is_none());

        let bundled = offline::BundledBackend::open(&runtime).unwrap();
        bundled.initialize_profile().await.unwrap();
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let mut command = bundled.command();
        command
            .args(["serve", "--host", "127.0.0.1", "--port"])
            .arg(port.to_string())
            .arg("--local-services-only")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        let child = super::spawn_owned_child(&mut command).unwrap();
        let pid = child.id().unwrap();
        let backend = super::SharedBackend::new(tokio::sync::Mutex::new(BackendManager {
            jarvis: Some(ChildHandle { child }),
            bundled_root: Some(runtime),
            ..BackendManager::default()
        }));
        let result = super::wait_for_jarvis_health(
            super::JarvisHealthTarget::Packaged {
                runtime: &bundled,
                port,
                mode: offline::AttachedBackendMode::LocalServices,
            },
            std::time::Duration::from_secs(25),
            &backend,
        ).await;
        if compatible {
            assert!(matches!(result, JarvisStartResult::Ready), "{result:?}");
        } else {
            assert!(matches!(result, JarvisStartResult::IncompatiblePackagedApi), "{result:?}");
        }
        let mut child = backend.lock().await.jarvis.take().unwrap().child;
        let status = super::SharedStatus::new(tokio::sync::Mutex::new(SetupStatus::default()));
        let unrelated = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        unrelated.set_nonblocking(true).unwrap();
        let unrelated_port = unrelated.local_addr().unwrap().port();
        assert!(!super::try_attach_bundled_server(&bundled, unrelated_port, &backend, &status, true).await);
        assert_eq!(unrelated.accept().unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
        assert!(backend.lock().await.attached.is_none());

        assert!(super::try_attach_bundled_server(&bundled, port, &backend, &status, true).await);
        let identity = backend.lock().await.attached.clone().unwrap();
        assert_eq!(identity.pid, pid);
        let state = status.lock().await.clone();
        assert!(!state.model_ready);
        assert!(!state.requires_source);
        if compatible {
            assert_eq!(state.phase, "local_services_ready");
            assert!(state.server_ready);
            assert!(state.error.is_none());
        } else {
            assert_eq!(state.phase, "server");
            assert!(!state.server_ready);
            assert!(state.error.unwrap().contains("Return to local services"));
        }
        backend.lock().await.stop_children().await;
        assert!(child.try_wait().unwrap().is_none(), "Attached server was stopped.");

        let mut stale = identity.clone();
        stale.started_at.push('0');
        assert!(bundled.stop_attached_server(port, &stale).await.is_err());
        assert!(child.try_wait().unwrap().is_none(), "Stale identity stopped a server.");
        super::stop_attached_backend(&backend, port).await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(10), child.wait())
            .await
            .expect("Explicit recovery must stop the verified backend.")
            .unwrap();
        assert!(backend.lock().await.attached.is_none());
    }

    #[derive(Default)]
    struct MemoryCredentialStore {
        values: StdMutex<HashMap<String, String>>,
    }

    impl MemoryCredentialStore {
        fn put(&self, key_name: &str, key_value: &str) {
            self.values
                .lock()
                .unwrap()
                .insert(key_name.to_string(), key_value.to_string());
        }

        fn value(&self, key_name: &str) -> Option<String> {
            self.values.lock().unwrap().get(key_name).cloned()
        }
    }

    impl InferenceCredentialStore for MemoryCredentialStore {
        fn get(&self, key_name: &str) -> Result<Option<String>, String> {
            Ok(self.value(key_name))
        }

        fn set(&self, key_name: &str, key_value: &str) -> Result<(), String> {
            let mut values = self.values.lock().unwrap();
            if key_value.is_empty() {
                values.remove(key_name);
            } else {
                values.insert(key_name.to_string(), key_value.to_string());
            }
            Ok(())
        }
    }

    fn unique_test_root(label: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "openjarvis-{}-{}-{}",
            label,
            std::process::id(),
            nonce
        ))
    }

    #[test]
    fn project_root_installed_desktop_finds_checkout_beside_executable() {
        let root = unique_test_root("installed-desktop-root");
        let install_dir = root.join("localappdata").join("OpenJarvis");
        let bundled_checkout = install_dir.join("src");
        let unrelated_clone = root.join("home").join("OpenJarvis");
        std::fs::create_dir_all(&bundled_checkout).unwrap();
        std::fs::create_dir_all(&unrelated_clone).unwrap();
        std::fs::write(bundled_checkout.join("pyproject.toml"), "[project]\n").unwrap();
        std::fs::write(unrelated_clone.join("pyproject.toml"), "[project]\n").unwrap();

        let exe = install_dir.join("openjarvis-desktop.exe");
        assert_eq!(project_root_near_executable(&exe), Some(bundled_checkout));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn project_root_development_executable_still_finds_ancestor_checkout() {
        let root = unique_test_root("development-desktop-root");
        let exe_dir = root.join("frontend").join("src-tauri").join("target");
        std::fs::create_dir_all(&exe_dir).unwrap();
        std::fs::write(root.join("pyproject.toml"), "[project]\n").unwrap();

        let exe = exe_dir.join("openjarvis-desktop");
        assert_eq!(project_root_near_executable(&exe), Some(root.clone()));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn tail_returns_whole_string_when_shorter_than_limit() {
        assert_eq!(uv_sync_stderr_tail("short error", 800), "short error");
    }

    #[test]
    fn tail_keeps_the_end_not_the_beginning() {
        // uv's actionable line is at the end; the spinner noise is at the start.
        let s = format!("{}ACTUAL ERROR HERE", "spinner-noise ".repeat(200));
        let tail = uv_sync_stderr_tail(&s, 40);
        assert!(tail.ends_with("ACTUAL ERROR HERE"), "tail was: {tail:?}");
        assert!(!tail.contains("spinner-noise spinner-noise spinner-noise"));
        assert!(tail.chars().count() <= 40);
    }

    #[test]
    fn tail_trims_surrounding_whitespace() {
        assert_eq!(uv_sync_stderr_tail("  \n padded \n  ", 800), "padded");
    }

    #[test]
    fn tail_never_splits_a_multibyte_codepoint() {
        // Each "é" is 2 bytes / 1 char. A byte-based slice could panic or
        // produce invalid UTF-8; the char-based tail must not.
        let s = "é".repeat(500);
        let tail = uv_sync_stderr_tail(&s, 100);
        assert_eq!(tail.chars().count(), 100);
        assert!(tail.chars().all(|c| c == 'é'));
    }

    #[test]
    fn failure_message_includes_exit_code_and_tail_and_hint() {
        let msg = format_uv_sync_failure(
            Path::new("/home/u/.openjarvis/src"),
            Some(2),
            "error: failed to resolve numpy==2.1.3",
        );
        assert!(msg.contains("exit 2"));
        assert!(msg.contains("/home/u/.openjarvis/src"));
        assert!(msg.contains("failed to resolve numpy==2.1.3"));
        assert!(msg.contains(DESKTOP_UV_SYNC_COMMAND)); // actionable next step
    }

    #[test]
    fn failure_message_renders_missing_exit_code_as_unknown() {
        // Process killed by signal → no exit code. Must not show a misleading -1.
        let msg = format_uv_sync_failure(Path::new("/x"), None, "boom");
        assert!(msg.contains("exit unknown"));
        assert!(!msg.contains("exit -1"));
    }

    #[test]
    fn spawn_error_names_the_binary_and_root() {
        let msg = format_uv_sync_spawn_error(
            Path::new("/repo"),
            "C:\\Users\\me\\.local\\bin\\uv.exe",
            "No such file or directory (os error 2)",
        );
        assert!(msg.contains("C:\\Users\\me\\.local\\bin\\uv.exe"));
        assert!(msg.contains("/repo"));
        assert!(msg.contains("No such file or directory"));
    }

    #[test]
    fn missing_rust_toolchain_message_names_cargo_and_installer() {
        let msg = format_missing_rust_toolchain();
        assert!(msg.contains("cargo"));
        assert!(msg.contains("https://rustup.rs"));
        assert!(msg.contains("openjarvis_rust"));
        assert!(msg.contains("Visual Studio Build Tools"));
    }

    #[test]
    fn uv_sync_rust_failure_mentions_toolchain() {
        let msg = format_uv_sync_failure(
            Path::new("C:\\Users\\me\\OpenJarvis"),
            Some(1),
            "maturin failed: linker `link.exe` not found while building openjarvis-rust",
        );
        assert!(msg.contains("exit 1"));
        assert!(msg.contains("link.exe"));
        assert!(msg.contains("https://rustup.rs"));
        assert!(msg.contains("Visual Studio Build Tools"));
    }

    #[test]
    fn extension_import_failure_names_verification_command() {
        let msg = format_extension_import_failure(
            Path::new("C:\\Users\\me\\OpenJarvis"),
            "ModuleNotFoundError: No module named 'openjarvis_rust'",
        );
        assert!(msg.contains("openjarvis_rust"));
        assert!(msg.contains(DESKTOP_UV_SYNC_COMMAND));
        assert!(msg.contains("uv run python -c \"import openjarvis_rust\""));
        assert!(msg.contains("ModuleNotFoundError"));
    }

    #[test]
    fn port_unavailable_message_names_port_and_owner_hint() {
        let msg = format_port_unavailable(8000, "address already in use");
        assert!(msg.contains("Port 8000 is not available"));
        assert!(msg.contains("address already in use"));
        assert!(msg.contains("To identify it"));
        assert!(msg.contains("8000"));
    }

    #[test]
    fn ollama_model_tag_validation_trims_and_rejects_unsafe_or_empty_tags() {
        assert_eq!(
            normalize_ollama_model_tag(" qwen3.5:4b ").unwrap(),
            "qwen3.5:4b"
        );
        assert!(normalize_ollama_model_tag("").is_err());
        assert!(normalize_ollama_model_tag("bad tag").is_err());
        assert!(normalize_ollama_model_tag("../model").is_err());
        assert!(normalize_ollama_model_tag(&"a".repeat(129)).is_err());
    }

    #[test]
    fn parse_ollama_model_names_reads_nonempty_names() {
        let body = serde_json::json!({
            "models": [
                {"name": "llama3.2:latest"},
                {"name": ""},
                {"name": "qwen3.5:4b"},
                {"model": "mistral:latest"}
            ]
        });
        assert_eq!(
            parse_ollama_model_names(&body),
            vec![
                "llama3.2:latest".to_string(),
                "qwen3.5:4b".to_string(),
                "mistral:latest".to_string()
            ]
        );
    }

    #[test]
    fn parse_ollama_models_keeps_names_and_sizes_for_settings() {
        let body = serde_json::json!({
            "models": [
                {"name": "qwen3.5:4b", "size": 4_200_000_000_u64},
                {"name": "blank", "size": null},
                {"name": "", "size": 100}
            ]
        });
        assert_eq!(
            parse_ollama_models(&body),
            vec![
                serde_json::json!({"name": "qwen3.5:4b", "size": 4_200_000_000_u64}),
                serde_json::json!({"name": "blank", "size": 0})
            ]
        );
    }

    #[test]
    fn model_names_match_treats_latest_as_optional() {
        assert!(model_names_match("llama3.2:latest", "llama3.2"));
        assert!(model_names_match("llama3.2", "llama3.2:latest"));
        assert!(model_names_match("qwen3.5:4b", "qwen3.5:4b"));
        assert!(!model_names_match("llama3.2:latest", "qwen3.5:4b"));
    }

    #[test]
    fn installed_model_resolution_only_uses_the_selected_tag() {
        let models = vec!["llama3.2:latest".to_string(), "qwen3.5:4b".to_string()];
        assert_eq!(
            matching_installed_model(&models, "llama3.2"),
            Some("llama3.2:latest".to_string())
        );
        assert_eq!(matching_installed_model(&models, "missing:3b"), None);
    }

    #[test]
    fn startup_requires_an_explicit_valid_source_config() {
        assert!(parse_configured_inference_config("").is_none());
        assert!(parse_configured_inference_config("not json").is_none());
        assert!(parse_configured_inference_config("{}").is_none());

        let ollama = parse_configured_inference_config(r#"{"kind":"ollama"}"#).unwrap();
        assert!(matches!(ollama.kind, SourceKind::Ollama));
        assert!(ollama.confirmed, "legacy explicit configs remain trusted");

        let custom = parse_configured_inference_config(
            r#"{"kind":"custom","model":"m","host":"http://localhost:1234"}"#,
        )
        .unwrap();
        assert!(matches!(custom.kind, SourceKind::Custom));
        assert!(custom.confirmed);

        let pending = parse_configured_inference_config(
            r#"{"kind":"custom","confirmed":false,"model":"m","host":"http://localhost:1234"}"#,
        )
        .unwrap();
        assert!(!pending.confirmed);
    }

    #[test]
    fn initial_setup_status_is_inert_until_a_source_is_chosen() {
        let status = SetupStatus::default();
        assert!(status.requires_source);
        assert_eq!(status.phase, "awaiting_source");
        assert_eq!(status.source, "unconfigured");
        assert!(!status.ollama_ready);
        assert!(!status.model_ready);
        assert!(!status.server_ready);
    }

    #[test]
    fn persisted_source_status_can_enter_startup() {
        let cfg = InferenceConfig {
            kind: SourceKind::Custom,
            confirmed: true,
            model: Some("m".into()),
            host: Some("http://localhost:1234".into()),
            engine: Some("lmstudio".into()),
        };
        let status = SetupStatus::starting(&cfg);
        assert!(!status.requires_source);
        assert_eq!(status.phase, "starting");
        assert_eq!(status.source, "custom");
    }

    #[test]
    fn parse_reads_custom_endpoint() {
        let cfg = parse_configured_inference_config(
            r#"{"kind":"custom","model":"qwen2.5-7b","host":"http://localhost:1234","engine":"lmstudio"}"#,
        )
        .unwrap();
        assert!(matches!(cfg.kind, SourceKind::Custom));
        assert_eq!(cfg.model.as_deref(), Some("qwen2.5-7b"));
        assert_eq!(cfg.host.as_deref(), Some("http://localhost:1234"));
        assert_eq!(cfg.engine.as_deref(), Some("lmstudio"));
    }

    #[test]
    fn normalize_host_strips_trailing_slash_and_v1() {
        assert_eq!(
            normalize_host("http://localhost:1234/v1"),
            "http://localhost:1234"
        );
        assert_eq!(
            normalize_host("http://localhost:1234/v1/"),
            "http://localhost:1234"
        );
        assert_eq!(
            normalize_host("http://localhost:1234/"),
            "http://localhost:1234"
        );
        assert_eq!(normalize_host("http://host:8000"), "http://host:8000");
    }

    #[test]
    fn boot_plan_ollama_requires_a_user_selected_model() {
        let cfg = InferenceConfig {
            kind: SourceKind::Ollama,
            ..Default::default()
        };
        let error = boot_plan(&cfg).unwrap_err();
        assert!(error.contains("Choose an Ollama model tag in Settings"));
    }

    #[test]
    fn boot_plan_ollama_launches_and_pulls_the_exact_selected_model() {
        let cfg = InferenceConfig {
            kind: SourceKind::Ollama,
            model: Some("qwen3.5:4b".into()),
            ..Default::default()
        };
        let plan = boot_plan(&cfg).unwrap();
        assert!(plan.launch_ollama);
        assert_eq!(plan.model_to_pull.as_deref(), Some("qwen3.5:4b"));
        assert!(plan.engine_host.is_none());
        assert!(plan
            .serve_args
            .windows(2)
            .any(|w| w == ["--engine", "ollama"]));
        assert!(plan
            .serve_args
            .windows(2)
            .any(|w| w == ["--model", "qwen3.5:4b"]));
    }

    #[test]
    fn boot_plan_ollama_respects_pinned_model() {
        let cfg = InferenceConfig {
            kind: SourceKind::Ollama,
            model: Some("qwen3.5:9b".into()),
            ..Default::default()
        };
        let plan = boot_plan(&cfg).unwrap();
        assert_eq!(plan.model_to_pull.as_deref(), Some("qwen3.5:9b"));
    }

    #[test]
    fn boot_plan_custom_skips_ollama_and_sets_engine_host() {
        let cfg = InferenceConfig {
            kind: SourceKind::Custom,
            confirmed: true,
            model: Some("qwen2.5-7b".into()),
            host: Some("http://localhost:1234".into()),
            engine: Some("lmstudio".into()),
        };
        let plan = boot_plan(&cfg).unwrap();
        assert!(!plan.launch_ollama);
        assert!(plan.model_to_pull.is_none());
        assert_eq!(
            plan.engine_host,
            Some(("lmstudio".to_string(), "http://localhost:1234".to_string()))
        );
        assert!(plan
            .serve_args
            .windows(2)
            .any(|w| w == ["--engine", "lmstudio"]));
        assert!(plan
            .serve_args
            .windows(2)
            .any(|w| w == ["--model", "qwen2.5-7b"]));
    }

    #[test]
    fn boot_plan_custom_defaults_engine_to_lmstudio() {
        let cfg = InferenceConfig {
            kind: SourceKind::Custom,
            confirmed: true,
            model: Some("m".into()),
            host: Some("http://h:1".into()),
            engine: None,
        };
        let plan = boot_plan(&cfg).unwrap();
        assert_eq!(plan.engine_host.as_ref().unwrap().0, "lmstudio");
        assert!(plan
            .serve_args
            .windows(2)
            .any(|w| w == ["--engine", "lmstudio"]));
    }

    #[test]
    fn boot_plan_custom_omits_engine_host_when_no_host() {
        // No configured host → don't set engine_host (no override to write).
        let cfg = InferenceConfig {
            kind: SourceKind::Custom,
            confirmed: true,
            model: Some("m".into()),
            host: None,
            engine: Some("lmstudio".into()),
        };
        let plan = boot_plan(&cfg).unwrap();
        assert!(plan.engine_host.is_none());
    }

    #[test]
    fn pending_config_rolls_back_unless_boot_confirms_it() {
        let root = std::env::temp_dir().join(format!(
            "openjarvis-pending-inference-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let pending_path = root.join("pending.json");
        std::fs::write(&pending_path, "pending").unwrap();
        let pending = InferenceConfig {
            kind: SourceKind::Custom,
            confirmed: false,
            model: Some("m".into()),
            host: Some("http://localhost:1".into()),
            engine: Some("lmstudio".into()),
        };

        {
            let _rollback = PendingInferenceRollback::at_path(&pending, pending_path.clone());
        }
        assert!(!pending_path.exists());

        let confirmed_path = root.join("confirmed.json");
        std::fs::write(&confirmed_path, "confirmed").unwrap();
        let mut confirmed = pending;
        confirmed.confirmed = true;
        {
            let _rollback = PendingInferenceRollback::at_path(&confirmed, confirmed_path.clone());
        }
        assert!(confirmed_path.exists());

        let committed_path = root.join("committed.json");
        std::fs::write(&committed_path, "pending").unwrap();
        let mut staged = InferenceConfig {
            kind: SourceKind::Ollama,
            confirmed: false,
            model: Some("qwen3.5:4b".into()),
            host: None,
            engine: None,
        };
        {
            let mut rollback = PendingInferenceRollback::at_path(&staged, committed_path.clone());
            rollback.confirm(&mut staged).unwrap();
        }
        assert!(staged.confirmed);
        assert!(committed_path.exists());
        let committed: InferenceConfig =
            serde_json::from_str(&std::fs::read_to_string(&committed_path).unwrap()).unwrap();
        assert!(committed.confirmed);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn pending_custom_key_is_staged_without_overwriting_existing_key() {
        let root = unique_test_root("stage-inference-key");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("inference.json");
        let store = MemoryCredentialStore::default();
        store.put("LMSTUDIO_API_KEY", "existing-key");
        let cfg = InferenceConfig {
            kind: SourceKind::Custom,
            confirmed: false,
            model: Some("m".into()),
            host: Some("http://localhost:1234".into()),
            engine: Some("lmstudio".into()),
        };

        stage_pending_inference_config_at(&cfg, Some("new-key"), &store, &path).unwrap();

        assert_eq!(
            store.value("LMSTUDIO_API_KEY").as_deref(),
            Some("existing-key")
        );
        assert_eq!(
            store.value(PENDING_INFERENCE_API_KEY).as_deref(),
            Some("new-key")
        );
        let staged: InferenceConfig =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(!staged.confirmed);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cancelled_pending_setup_discards_stage_and_preserves_existing_key() {
        let root = unique_test_root("cancel-inference-key");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("inference.json");
        let store = std::sync::Arc::new(MemoryCredentialStore::default());
        store.put("LMSTUDIO_API_KEY", "existing-key");
        let cfg = InferenceConfig {
            kind: SourceKind::Custom,
            confirmed: false,
            model: Some("m".into()),
            host: Some("http://localhost:1234".into()),
            engine: Some("lmstudio".into()),
        };
        stage_pending_inference_config_at(&cfg, Some("new-key"), store.as_ref(), &path).unwrap();

        {
            let _rollback =
                PendingInferenceRollback::at_path_with_store(&cfg, path.clone(), store.clone());
        }

        assert!(!path.exists());
        assert_eq!(store.value(PENDING_INFERENCE_API_KEY), None);
        assert_eq!(
            store.value("LMSTUDIO_API_KEY").as_deref(),
            Some("existing-key")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn successful_pending_setup_promotes_stage_only_at_confirmation() {
        let root = unique_test_root("confirm-inference-key");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("inference.json");
        let store = std::sync::Arc::new(MemoryCredentialStore::default());
        store.put("LMSTUDIO_API_KEY", "existing-key");
        let mut cfg = InferenceConfig {
            kind: SourceKind::Custom,
            confirmed: false,
            model: Some("m".into()),
            host: Some("http://localhost:1234".into()),
            engine: Some("lmstudio".into()),
        };
        stage_pending_inference_config_at(&cfg, Some("new-key"), store.as_ref(), &path).unwrap();

        {
            let mut rollback =
                PendingInferenceRollback::at_path_with_store(&cfg, path.clone(), store.clone());
            rollback.confirm(&mut cfg).unwrap();
        }

        assert!(cfg.confirmed);
        assert_eq!(store.value(PENDING_INFERENCE_API_KEY), None);
        assert_eq!(store.value("LMSTUDIO_API_KEY").as_deref(), Some("new-key"));
        let confirmed: InferenceConfig =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(confirmed.confirmed);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_confirmation_restores_existing_key_and_leaves_no_stage() {
        let root = unique_test_root("failed-confirm-inference-key");
        let path = root.join("config-is-a-directory");
        std::fs::create_dir_all(&path).unwrap();
        let store = std::sync::Arc::new(MemoryCredentialStore::default());
        store.put("LMSTUDIO_API_KEY", "existing-key");
        store.put(PENDING_INFERENCE_API_KEY, "new-key");
        let mut cfg = InferenceConfig {
            kind: SourceKind::Custom,
            confirmed: false,
            model: Some("m".into()),
            host: Some("http://localhost:1234".into()),
            engine: Some("lmstudio".into()),
        };

        let result = {
            let mut rollback =
                PendingInferenceRollback::at_path_with_store(&cfg, path.clone(), store.clone());
            rollback.confirm(&mut cfg)
        };

        assert!(result.is_err());
        assert!(!cfg.confirmed);
        assert_eq!(store.value(PENDING_INFERENCE_API_KEY), None);
        assert_eq!(
            store.value("LMSTUDIO_API_KEY").as_deref(),
            Some("existing-key")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_pending_config_write_discards_new_stage() {
        let root = unique_test_root("failed-stage-inference-key");
        let path = root.join("config-is-a-directory");
        std::fs::create_dir_all(&path).unwrap();
        let store = MemoryCredentialStore::default();
        store.put("LMSTUDIO_API_KEY", "existing-key");
        let cfg = InferenceConfig {
            kind: SourceKind::Custom,
            confirmed: false,
            model: Some("m".into()),
            host: Some("http://localhost:1234".into()),
            engine: Some("lmstudio".into()),
        };

        let result = stage_pending_inference_config_at(&cfg, Some("new-key"), &store, &path);

        assert!(result.is_err());
        assert_eq!(store.value(PENDING_INFERENCE_API_KEY), None);
        assert_eq!(
            store.value("LMSTUDIO_API_KEY").as_deref(),
            Some("existing-key")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_confirmed_config_write_restores_existing_key() {
        let root = unique_test_root("failed-confirmed-inference-key");
        let path = root.join("config-is-a-directory");
        std::fs::create_dir_all(&path).unwrap();
        let store = MemoryCredentialStore::default();
        store.put("LMSTUDIO_API_KEY", "existing-key");
        let cfg = InferenceConfig {
            kind: SourceKind::Custom,
            confirmed: true,
            model: Some("m".into()),
            host: Some("http://localhost:1234".into()),
            engine: Some("lmstudio".into()),
        };

        let result = persist_confirmed_inference_config_at(&cfg, Some("new-key"), &store, &path);

        assert!(result.is_err());
        assert_eq!(
            store.value("LMSTUDIO_API_KEY").as_deref(),
            Some("existing-key")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reset_discards_pending_config_and_stage_without_touching_existing_key() {
        let root = unique_test_root("reset-inference-key");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("inference.json");
        std::fs::write(&path, "pending").unwrap();
        let store = MemoryCredentialStore::default();
        store.put("LMSTUDIO_API_KEY", "existing-key");
        store.put(PENDING_INFERENCE_API_KEY, "new-key");

        discard_pending_inference_setup_at(&path, &store).unwrap();

        assert!(!path.exists());
        assert_eq!(store.value(PENDING_INFERENCE_API_KEY), None);
        assert_eq!(
            store.value("LMSTUDIO_API_KEY").as_deref(),
            Some("existing-key")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn ready_port_squatter_never_receives_cloud_key_without_owned_child() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let reload_url = format!("http://{}/v1/cloud/reload", listener.local_addr().unwrap());
        let backend = std::sync::Arc::new(tokio::sync::Mutex::new(BackendManager::default()));
        let mut setup_status = SetupStatus::default();
        setup_status.server_ready = true;
        let status = std::sync::Arc::new(tokio::sync::Mutex::new(setup_status));

        reload_cloud_keys_for_owned_backend_at(
            &backend,
            &status,
            vec![("OPENAI_API_KEY".into(), "must-not-leak".into())],
            &reload_url,
        )
        .await;

        let error = listener.accept().unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::WouldBlock);
    }

    #[tokio::test]
    async fn stopping_backend_aborts_the_tracked_boot_task() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        struct DropSignal(Arc<AtomicBool>);
        impl Drop for DropSignal {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let signal = DropSignal(dropped.clone());
        let task = tokio::spawn(async move {
            let _signal = signal;
            std::future::pending::<()>().await;
        });
        tokio::task::yield_now().await;

        let mut manager = BackendManager::default();
        manager.boot_task = Some(tauri::async_runtime::JoinHandle::Tokio(task));
        manager.stop_all().await;

        assert!(manager.boot_task.is_none());
        assert!(
            dropped.load(Ordering::SeqCst),
            "stop_all must not return before the aborted boot future is dropped"
        );
    }

    #[test]
    fn owned_child_process_helper() {
        let Some(started) = std::env::var_os("OPENJARVIS_OWNED_CHILD_STARTED") else {
            return;
        };
        let completed = std::env::var_os("OPENJARVIS_OWNED_CHILD_COMPLETED").unwrap();
        std::fs::write(started, "started").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(500));
        std::fs::write(completed, "completed").unwrap();
    }

    #[tokio::test]
    async fn aborting_boot_kills_child_before_manager_registration() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "openjarvis-owned-child-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let started = root.join("started");
        let completed = root.join("completed");

        let (spawned_tx, spawned_rx) = tokio::sync::oneshot::channel();
        let started_for_child = started.clone();
        let completed_for_child = completed.clone();
        let task = tokio::spawn(async move {
            let mut cmd = tokio::process::Command::new(std::env::current_exe().unwrap());
            cmd.args([
                "--exact",
                "tests::owned_child_process_helper",
                "--nocapture",
            ])
            .env("OPENJARVIS_OWNED_CHILD_STARTED", started_for_child)
            .env("OPENJARVIS_OWNED_CHILD_COMPLETED", completed_for_child)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
            let child = spawn_owned_child(&mut cmd).unwrap();
            spawned_tx.send(()).unwrap();
            let _unregistered_child = child;
            std::future::pending::<()>().await;
        });
        spawned_rx.await.unwrap();

        for _ in 0..100 {
            if started.exists() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(started.exists(), "helper process did not start");

        task.abort();
        let _ = task.await;
        tokio::time::sleep(std::time::Duration::from_millis(700)).await;

        assert!(
            !completed.exists(),
            "an aborted boot task left its not-yet-registered child running"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn boot_plan_ollama_keeps_the_choice_independent_of_hardware() {
        let cfg = InferenceConfig {
            kind: SourceKind::Ollama,
            model: Some("qwen3.5:35b".into()),
            ..Default::default()
        };
        let plan = boot_plan(&cfg).unwrap();
        assert_eq!(plan.model_to_pull.as_deref(), Some("qwen3.5:35b"));
    }

    #[test]
    fn upsert_engine_host_writes_into_empty_config() {
        let out = upsert_engine_host("", "lmstudio", "http://localhost:1234").unwrap();
        let doc: toml_edit::DocumentMut = out.parse().unwrap();
        assert_eq!(
            doc["engine"]["lmstudio"]["host"].as_str(),
            Some("http://localhost:1234")
        );
    }

    #[test]
    fn upsert_engine_host_preserves_existing_content() {
        let existing = "[intelligence]\ndefault_model = \"keep-me\"\n";
        let out = upsert_engine_host(existing, "vllm", "http://host:8000").unwrap();
        let doc: toml_edit::DocumentMut = out.parse().unwrap();
        assert_eq!(
            doc["intelligence"]["default_model"].as_str(),
            Some("keep-me")
        );
        assert_eq!(
            doc["engine"]["vllm"]["host"].as_str(),
            Some("http://host:8000")
        );
    }

    #[test]
    fn upsert_engine_host_updates_existing_host() {
        let existing = "[engine.lmstudio]\nhost = \"http://old:1\"\n";
        let out = upsert_engine_host(existing, "lmstudio", "http://new:2").unwrap();
        let doc: toml_edit::DocumentMut = out.parse().unwrap();
        assert_eq!(
            doc["engine"]["lmstudio"]["host"].as_str(),
            Some("http://new:2")
        );
    }

    // -----------------------------------------------------------------
    // #455 — AppImage subprocess env-strip helper
    // -----------------------------------------------------------------
    //
    // `prepare_subprocess_for_appimage` strips LD_LIBRARY_PATH (and the
    // related AppImage runtime variables) from a child `Command` ONLY
    // when the parent process is itself running inside an AppImage —
    // detected by the presence of the `APPIMAGE` env variable that the
    // AppImage runtime sets to the original .AppImage path. We can't
    // observe the env_remove calls directly through tokio's Command
    // API (it doesn't expose its env map publicly), so these tests
    // exercise the documented contract on each platform:
    //
    //   * on macOS / Windows: the function is a no-op regardless of env.
    //   * on Linux without $APPIMAGE: also a no-op.
    //   * on Linux with $APPIMAGE: it doesn't panic, doesn't return an
    //     error, and the calling code that follows succeeds. The
    //     observable behaviour test is the integration repro on a real
    //     AppImage build (covered in PR test plan).
    //
    // The Mutex serialises any test that touches the process-wide
    // `APPIMAGE` env var so cargo test's parallel runner can't race two
    // tests setting and unsetting it concurrently. `static Mutex` works
    // on a const path since Rust 1.63 (and Tauri's MSRV is well above).

    static APPIMAGE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // Pick a binary that exists on every test target so the test body is
    // doing something other than constructing an obviously-broken command
    // path on Windows.
    #[cfg(target_os = "windows")]
    const HARMLESS_BIN: &str = "cmd";
    #[cfg(not(target_os = "windows"))]
    const HARMLESS_BIN: &str = "/bin/true";

    #[test]
    fn prepare_subprocess_for_appimage_no_appimage_is_safe() {
        let _guard = APPIMAGE_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("APPIMAGE");
        // SAFETY: APPIMAGE_ENV_LOCK serialises every test that touches
        // this env var, so the mutation is single-threaded for the
        // duration of the lock. The 2024-edition env mutation rules
        // require the `unsafe` block but the guard makes it sound.
        unsafe {
            std::env::remove_var("APPIMAGE");
        }
        let mut cmd = tokio::process::Command::new(HARMLESS_BIN);
        super::prepare_subprocess_for_appimage(&mut cmd);
        if let Some(v) = prev {
            unsafe {
                std::env::set_var("APPIMAGE", v);
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn prepare_subprocess_for_appimage_with_appimage_set_is_safe() {
        let _guard = APPIMAGE_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("APPIMAGE");
        unsafe {
            std::env::set_var("APPIMAGE", "/tmp/test.AppImage");
        }
        let mut cmd = tokio::process::Command::new(HARMLESS_BIN);
        super::prepare_subprocess_for_appimage(&mut cmd);
        unsafe {
            if let Some(v) = prev {
                std::env::set_var("APPIMAGE", v);
            } else {
                std::env::remove_var("APPIMAGE");
            }
        }
    }
}
