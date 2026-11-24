//! Desktop-only access to the pair of credentials used by the Cloudflare worker.

use super::{cloudflare_launch_environment, CloudflareEnvironment, CLOUDFLARE_ENV};

const STORAGE_ERROR: &str =
    "Could not access Cloudflare credentials in your system credential store.";
const INVALID_RECORD: &str =
    "The saved Cloudflare credentials are invalid. Enter your Account ID and API token again.";

#[derive(serde::Deserialize, serde::Serialize)]
struct Credentials {
    account_id: String,
    api_token: String,
}

impl Credentials {
    fn validated(account_id: &str, api_token: &str) -> Result<Self, String> {
        let account_id = account_id.trim();
        let api_token = api_token.trim();
        if account_id.len() != 32 || !account_id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("Enter the 32-character Cloudflare Account ID.".into());
        }
        // Workers AI accepts printable ASCII without spaces. Bound the stored
        // record to the size supported by the Windows credential store.
        if api_token.is_empty()
            || api_token.len() > 512
            || !api_token.bytes().all(|byte| byte.is_ascii_graphic())
        {
            return Err("Enter only the Cloudflare Workers AI API token.".into());
        }
        Ok(Self {
            account_id: account_id.into(),
            api_token: api_token.into(),
        })
    }

    fn into_environment(self) -> CloudflareEnvironment {
        [
            (CLOUDFLARE_ENV[0], self.account_id),
            (CLOUDFLARE_ENV[1], self.api_token),
        ]
    }
}

#[derive(serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum CredentialStatus {
    Stored,
    Environment,
    Missing,
}

enum AvailableCredentials {
    Stored(Credentials),
    Environment(Credentials),
    Missing,
}

fn entry() -> Result<keyring::Entry, String> {
    keyring::Entry::new("A.C.E. Cloudflare Workers AI", "credentials")
        .map_err(|_| STORAGE_ERROR.into())
}

fn read(entry: &keyring::Entry) -> Result<Option<Credentials>, String> {
    match entry.get_password() {
        Ok(value) => {
            let parsed: Credentials =
                serde_json::from_str(&value).map_err(|_| INVALID_RECORD.to_string())?;
            Credentials::validated(&parsed.account_id, &parsed.api_token)
                .map(Some)
                .map_err(|_| INVALID_RECORD.into())
        }
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(_) => Err(STORAGE_ERROR.into()),
    }
}

fn write(entry: &keyring::Entry, credentials: &Credentials) -> Result<(), String> {
    // Replace both values together. A failed write cannot leave a mixed pair.
    let value = serde_json::to_string(credentials).map_err(|_| STORAGE_ERROR.to_string())?;
    entry.set_password(&value).map_err(|_| STORAGE_ERROR.into())
}

fn remove(entry: &keyring::Entry) -> Result<(), String> {
    match entry.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(_) => Err(STORAGE_ERROR.into()),
    }
}

fn choose_credentials(
    stored: Result<Option<Credentials>, String>,
    from_environment: impl FnOnce() -> Option<Credentials>,
) -> Result<AvailableCredentials, String> {
    match stored {
        Ok(Some(credentials)) => Ok(AvailableCredentials::Stored(credentials)),
        other => match from_environment() {
            Some(credentials) => Ok(AvailableCredentials::Environment(credentials)),
            None => other.map(|_| AvailableCredentials::Missing),
        },
    }
}

fn available_credentials() -> Result<AvailableCredentials, String> {
    choose_credentials(entry().and_then(|entry| read(&entry)), || {
        let [(_, account_id), (_, api_token)] = cloudflare_launch_environment().ok()?;
        Credentials::validated(&account_id, &api_token).ok()
    })
}

async fn access_credentials<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|_| STORAGE_ERROR.to_string())?
}

pub(super) async fn environment() -> Result<CloudflareEnvironment, String> {
    access_credentials(|| match available_credentials()? {
        AvailableCredentials::Stored(credentials)
        | AvailableCredentials::Environment(credentials) => Ok(credentials.into_environment()),
        AvailableCredentials::Missing => {
            Err("Enter your Cloudflare Account ID and Workers AI API token in A.C.E.".into())
        }
    })
    .await
}

fn require_main_webview(webview: &tauri::Webview) -> Result<(), String> {
    // Tauri keeps these application commands local to the app origin. Also
    // limit credential management to the main webview.
    if webview.label() != "main" {
        return Err("Manage Cloudflare credentials from the main A.C.E. window.".into());
    }
    Ok(())
}

#[tauri::command]
pub(super) async fn get_cloudflare_credentials_status(
    webview: tauri::Webview,
) -> Result<CredentialStatus, String> {
    require_main_webview(&webview)?;
    access_credentials(|| {
        Ok(match available_credentials()? {
            AvailableCredentials::Stored(_) => CredentialStatus::Stored,
            AvailableCredentials::Environment(_) => CredentialStatus::Environment,
            AvailableCredentials::Missing => CredentialStatus::Missing,
        })
    })
    .await
}

#[tauri::command]
pub(super) async fn save_cloudflare_credentials(
    webview: tauri::Webview,
    account_id: String,
    api_token: String,
) -> Result<(), String> {
    require_main_webview(&webview)?;
    let credentials = Credentials::validated(&account_id, &api_token)?;
    access_credentials(move || write(&entry()?, &credentials)).await
}

#[tauri::command]
pub(super) async fn remove_cloudflare_credentials(webview: tauri::Webview) -> Result<(), String> {
    require_main_webview(&webview)?;
    access_credentials(|| remove(&entry()?)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACCOUNT: &str = "0123456789abcdef0123456789abcdef";
    const TOKEN: &str = "synthetic-test-token";

    #[test]
    fn validates_and_trims_values_for_the_cloudflare_worker() {
        let credentials =
            Credentials::validated(&format!(" {ACCOUNT}\n"), &format!(" {TOKEN} ")).unwrap();
        assert_eq!(credentials.account_id, ACCOUNT);
        assert_eq!(credentials.api_token, TOKEN);
        assert!(Credentials::validated("not-an-account", TOKEN).is_err());
        for invalid in ["", "Bearer token", "\r\nfoo\nbar", "🚫", &"t".repeat(513)] {
            assert!(Credentials::validated(ACCOUNT, invalid).is_err());
        }
    }

    #[test]
    fn saves_replaces_and_removes_a_single_pair() {
        let entry =
            keyring::Entry::new_with_credential(Box::new(keyring::mock::MockCredential::default()));
        assert!(read(&entry).unwrap().is_none());
        write(&entry, &Credentials::validated(ACCOUNT, TOKEN).unwrap()).unwrap();
        assert_eq!(read(&entry).unwrap().unwrap().api_token, TOKEN);

        let replacement = Credentials::validated(&"f".repeat(32), "replacement").unwrap();
        write(&entry, &replacement).unwrap();
        let restored = read(&entry).unwrap().unwrap();
        assert_eq!(restored.account_id, replacement.account_id);
        assert_eq!(restored.api_token, replacement.api_token);
        remove(&entry).unwrap();
        remove(&entry).unwrap();
        assert!(read(&entry).unwrap().is_none());
    }

    #[test]
    fn prefers_saved_values_and_falls_back_to_a_complete_environment_pair() {
        let saved = Credentials::validated(ACCOUNT, TOKEN).unwrap();
        let picked = choose_credentials(Ok(Some(saved)), || {
            panic!("read environment over saved pair")
        })
        .unwrap();
        assert!(matches!(picked, AvailableCredentials::Stored(_)));

        let env = || Credentials::validated(ACCOUNT, TOKEN).ok();
        assert!(matches!(
            choose_credentials(Ok(None), env).unwrap(),
            AvailableCredentials::Environment(_)
        ));
        assert!(matches!(
            choose_credentials(Err(STORAGE_ERROR.into()), env).unwrap(),
            AvailableCredentials::Environment(_)
        ));
        assert!(matches!(
            choose_credentials(Ok(None), || None).unwrap(),
            AvailableCredentials::Missing
        ));
        assert!(choose_credentials(Err(STORAGE_ERROR.into()), || None).is_err());
    }

    #[cfg(target_os = "windows")]
    #[test]
    #[ignore = "requires Windows Credential Manager access"]
    fn writes_reads_and_removes_a_unique_windows_credential() {
        let service = format!(
            "A.C.E. Credential Test {}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let test_entry = keyring::Entry::new(&service, "synthetic-test").unwrap();
        let credentials = Credentials::validated(ACCOUNT, TOKEN).unwrap();
        write(&test_entry, &credentials).unwrap();
        let read_result = read(&keyring::Entry::new(&service, "synthetic-test").unwrap());
        remove(&test_entry).unwrap();
        let restored = read_result.unwrap().unwrap();
        assert_eq!(restored.account_id, ACCOUNT);
        assert_eq!(restored.api_token, TOKEN);
        assert!(read(&test_entry).unwrap().is_none());
    }
}
