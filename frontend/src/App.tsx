import { useEffect, useState, useCallback, useRef } from 'react';
import { Routes, Route } from 'react-router';
import { Layout } from './components/Layout';
import { ChatPage } from './pages/ChatPage';
import { DashboardPage } from './pages/DashboardPage';
import { SettingsPage } from './pages/SettingsPage';
import { GetStartedPage } from './pages/GetStartedPage';
import { AgentsPage } from './pages/AgentsPage';
import { DataSourcesPage } from './pages/DataSourcesPage';
import { WebResearchPage } from './pages/WebResearchPage';
import { LogsPage } from './pages/LogsPage';
import { KnowledgeGraphPage } from './pages/KnowledgeGraphPage';
import { MemoryFactsPage } from './pages/MemoryFactsPage';
import { SharedChatsPage } from './pages/SharedChatsPage';
import { CodeImprovementsPage } from './pages/CodeImprovementsPage';
import { LocalMonitorPage } from './pages/LocalMonitorPage';
import { ScreenAnalysisPage } from './pages/ScreenAnalysisPage';
import { CommandPalette } from './components/CommandPalette';
import { SetupScreen } from './components/SetupScreen';
import { InferenceSourceSetup } from './components/InferenceSourceSetup';
import {
  checkPackagedBackendHealth,
  isOfflineDistribution,
  type DesktopReadyMode,
  type PackagedBackendHealth,
} from './lib/desktop-startup';
import { Toaster } from './components/ui/sonner';
import { useAppStore } from './lib/store';
import { fetchModels, fetchServerInfo, fetchSavings, submitSavings, isTauri, startBackend } from './lib/api';
import { OptInModal } from './components/OptInModal';
import { UpdateChecker } from './components/Desktop/UpdateChecker';
import { track, hashId } from './lib/analytics';
import { offerAnalyticsChoice } from './lib/analytics-invitation';

export default function App() {
  const [readyMode, setReadyMode] = useState<DesktopReadyMode | null>(() =>
    isTauri() ? null : 'inference',
  );
  const [choosingInference, setChoosingInference] = useState(false);
  const [localServicesDetail, setLocalServicesDetail] = useState('');
  const [canRetryModel, setCanRetryModel] = useState(false);
  const [retryingModel, setRetryingModel] = useState(false);
  const [retryError, setRetryError] = useState('');
  const [packagedHealth, setPackagedHealth] = useState<PackagedBackendHealth | null>(null);
  const setupDone = readyMode !== null;
  const localServicesOnly = readyMode === 'local-services';
  const offlineDistribution = isOfflineDistribution();
  const backendReachable = packagedHealth === null ? null : packagedHealth !== 'unreachable';
  const inferenceUnavailable = readyMode === 'inference' && (
    packagedHealth === 'inference_unavailable' || packagedHealth === 'local_services'
  );
  const chatUnavailable = inferenceUnavailable || backendReachable === false;
  const chooseInference = useCallback(() => setChoosingInference(true), []);
  const handleSetupReady = useCallback((mode: DesktopReadyMode, detail = '', retrySavedModel = false) => {
    setPackagedHealth(null);
    setReadyMode(mode);
    setCanRetryModel(mode === 'local-services' && retrySavedModel);
    if (mode === 'local-services') {
      setLocalServicesDetail(detail);
      const store = useAppStore.getState();
      store.setServerInfo(null);
      store.setModels([], { requireExactModel: true });
      store.setModelsLoading(false);
      return;
    }
    if (!localStorage.getItem('oj-setup-completed')) {
      localStorage.setItem('oj-setup-completed', '1');
      track('setup_completed', { preset: 'default' });
    }
  }, []);
  useEffect(() => {
    if (!offlineDistribution || !setupDone || choosingInference) return;
    let stopped = false;
    let checking = false;
    const check = async () => {
      if (checking) return;
      checking = true;
      const health = await checkPackagedBackendHealth();
      if (!stopped) setPackagedHealth(health);
      checking = false;
    };
    void check();
    const interval = window.setInterval(check, 5000);
    window.addEventListener('focus', check);
    return () => {
      stopped = true;
      window.clearInterval(interval);
      window.removeEventListener('focus', check);
    };
  }, [offlineDistribution, setupDone, choosingInference]);

  const retrySavedModel = async () => {
    if (retryingModel) return;
    setRetryingModel(true);
    setRetryError('');
    try {
      await startBackend();
      setPackagedHealth(null);
      setReadyMode(null);
    } catch {
      setRetryError('Could not restart the saved model. Check its settings and try again.');
    } finally {
      setRetryingModel(false);
    }
  };

  const prevModelRef = useRef<string>('');
  const setModels = useAppStore((s) => s.setModels);
  const setModelsLoading = useAppStore((s) => s.setModelsLoading);
  const selectedModel = useAppStore((s) => s.selectedModel);
  const serverInfo = useAppStore((s) => s.serverInfo);
  const cloudflareOnline = serverInfo?.engine === 'cloudflare';
  const localInferenceUnavailable = localServicesOnly || chatUnavailable || cloudflareOnline;
  const monitorLocalModelAvailable = !localInferenceUnavailable && !!serverInfo?.model &&
    ['ollama', 'lmstudio', 'llamacpp'].includes(serverInfo.engine);
  const setServerInfo = useAppStore((s) => s.setServerInfo);
  const setSavings = useAppStore((s) => s.setSavings);
  const settings = useAppStore((s) => s.settings);
  const commandPaletteOpen = useAppStore((s) => s.commandPaletteOpen);
  const setCommandPaletteOpen = useAppStore((s) => s.setCommandPaletteOpen);
  const optInEnabled = useAppStore((s) => s.optInEnabled);
  const optInDisplayName = useAppStore((s) => s.optInDisplayName);
  const optInEmail = useAppStore((s) => s.optInEmail);
  const optInAnonId = useAppStore((s) => s.optInAnonId);
  const optInModalSeen = useAppStore((s) => s.optInModalSeen);
  const optInModalOpen = useAppStore((s) => s.optInModalOpen);
  const setOptInModalOpen = useAppStore((s) => s.setOptInModalOpen);
  const markOptInModalSeen = useAppStore((s) => s.markOptInModalSeen);
  const savings = useAppStore((s) => s.savings);

  // Apply theme class to <html>
  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove('dark', 'light');
    if (settings.theme === 'dark') root.classList.add('dark');
    else if (settings.theme === 'light') root.classList.add('light');
  }, [settings.theme]);

  // Sync overlay conversations into the main app
  const importOverlay = useAppStore((s) => s.importOverlayConversation);
  useEffect(() => {
    if (!isTauri()) return;
    importOverlay();
    const interval = setInterval(importOverlay, 5000);
    return () => clearInterval(interval);
  }, [importOverlay]);

  useEffect(() => {
    // Desktop setup has already fetched models when inference is ready.
    if (isTauri()) return;
    fetchModels()
      .then(setModels)
      .catch(() => setModels([]))
      .finally(() => setModelsLoading(false));
  }, [setModels, setModelsLoading]);

  useEffect(() => {
    // Desktop setup reads the running backend before enabling the composer.
    if (!setupDone || isTauri()) return;
    let active = true;
    fetchServerInfo()
      .then((info) => { if (active) setServerInfo(info); })
      .catch(() => {});
    return () => { active = false; };
  }, [readyMode, setupDone, setServerInfo]);

  // Poll savings and optionally share to Supabase
  useEffect(() => {
    if (!setupDone) return;
    const refresh = () =>
      fetchSavings()
        .then((data) => {
          setSavings(data);
          if (!offlineDistribution && optInEnabled && optInDisplayName && data) {
            const claudeEntry = data.per_provider.find(
              (p) => p.provider === 'claude-fable-5',
            );
            const dollarSavings = claudeEntry ? claudeEntry.total_cost : 0;
            const energySaved = data.per_provider.reduce(
              (sum, p) => sum + (p.energy_wh || 0),
              0,
            );
            const flopsSaved = data.per_provider.reduce(
              (sum, p) => sum + (p.flops || 0),
              0,
            );
            submitSavings({
              anon_id: optInAnonId,
              display_name: optInDisplayName,
              email: optInEmail,
              total_calls: data.total_calls,
              total_tokens: data.total_tokens,
              dollar_savings: dollarSavings,
              energy_wh_saved: energySaved,
              flops_saved: flopsSaved,
              token_counting_version: data.token_counting_version ?? 1,
            });
          }
        })
        .catch(() => {});
    refresh();
    const interval = setInterval(refresh, 30000);
    return () => clearInterval(interval);
  }, [setupDone, offlineDistribution, optInEnabled, optInDisplayName, optInEmail, optInAnonId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Wait for desktop setup before fetching identity or marking the dialog seen.
  useEffect(() => offerAnalyticsChoice({
    ready: setupDone && !offlineDistribution,
    sharingModalSeen: optInModalSeen,
    open: () => setOptInModalOpen(true),
    markSharingModalSeen: markOptInModalSeen,
  }), [setupDone, offlineDistribution, optInModalSeen, setOptInModalOpen, markOptInModalSeen]);

  // Fire model_changed when the user switches models. First mount is
  // not a "change" — only emit when both prev and current are real and
  // differ.
  useEffect(() => {
    const prev = prevModelRef.current;
    const curr = selectedModel || '';
    prevModelRef.current = curr;
    if (!prev || !curr || prev === curr) return;
    void (async () => {
      const [fromHash, toHash] = await Promise.all([
        hashId(prev),
        hashId(curr),
      ]);
      track('model_changed', {
        from_model_hash: fromHash,
        to_model_hash: toHash,
      });
    })();
  }, [selectedModel]);

  // app_opened — one-shot per app launch, fires after analytics has had
  // a chance to initialize. platform + version are super-properties
  // registered in analytics.ts initAnalytics, so no per-call props needed.
  useEffect(() => {
    const t = setTimeout(() => {
      track('app_opened', {});
    }, 500);
    return () => clearTimeout(t);
  }, []);

  const toggleSystemPanel = useAppStore((s) => s.toggleSystemPanel);

  // Global keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        if (localServicesOnly) chooseInference();
        else if (!cloudflareOnline) setCommandPaletteOpen(!commandPaletteOpen);
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'i') {
        e.preventDefault();
        toggleSystemPanel();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [chooseInference, cloudflareOnline, commandPaletteOpen, localServicesOnly, setCommandPaletteOpen, toggleSystemPanel]);


  if (!setupDone) {
    return <SetupScreen onReady={handleSetupReady} />;
  }
  if (choosingInference) {
    return (
      <InferenceSourceSetup
        onCancel={() => setChoosingInference(false)}
        onStarted={() => {
          setChoosingInference(false);
          setCommandPaletteOpen(false);
          setPackagedHealth(null);
          setReadyMode(null);
        }}
      />
    );
  }

  return (
    <>
      <UpdateChecker />
      <Routes>
        <Route element={
          <Layout
            localServicesDetail={localServicesDetail}
            backendReachable={offlineDistribution ? backendReachable : undefined}
            inferenceUnavailable={inferenceUnavailable}
            onConfigureModel={localServicesOnly ? chooseInference : undefined}
            onRetryModel={localServicesOnly && canRetryModel ? retrySavedModel : undefined}
            retryingModel={retryingModel}
            retryError={retryError}
          />
        }>
          <Route index element={
            <ChatPage localServicesOnly={localServicesOnly} inferenceUnavailable={chatUnavailable} />
          } />
          <Route path="shared-chats" element={
            <SharedChatsPage localServicesOnly={localInferenceUnavailable} />
          } />
          <Route path="dashboard" element={<DashboardPage />} />
          <Route path="settings" element={
            <SettingsPage localServicesOnly={localServicesOnly} onConfigureModel={chooseInference} />
          } />
          <Route path="get-started" element={<GetStartedPage />} />
          <Route path="data-sources" element={<DataSourcesPage />} />
          <Route path="web-research" element={
            <WebResearchPage
              backendUnavailable={backendReachable === false}
              inferenceUnavailable={localInferenceUnavailable}
            />
          } />
          <Route path="knowledge-graph" element={<KnowledgeGraphPage />} />
          <Route path="memory-facts" element={<MemoryFactsPage />} />
          <Route path="code-improvements" element={<CodeImprovementsPage />} />
          <Route path="system-monitoring" element={<LocalMonitorPage localModelAvailable={monitorLocalModelAvailable} />} />
          <Route path="screen-analysis" element={
            <ScreenAnalysisPage
              backendUnavailable={backendReachable === false}
              localInferenceUnavailable={localInferenceUnavailable}
              cloudflareSelected={cloudflareOnline}
            />
          } />
          <Route path="agents" element={<AgentsPage />} />
          <Route path="logs" element={<LogsPage />} />
        </Route>
      </Routes>
      <Toaster position="bottom-right" />
      {!localServicesOnly && !cloudflareOnline && commandPaletteOpen && <CommandPalette />}
      {!offlineDistribution && optInModalOpen && (
        <OptInModal onClose={() => setOptInModalOpen(false)} />
      )}
    </>
  );
}
