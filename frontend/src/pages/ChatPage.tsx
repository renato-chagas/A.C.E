import { ChatArea } from '../components/Chat/ChatArea';
import { SystemPanel } from '../components/Chat/SystemPanel';
import { useAppStore } from '../lib/store';

export function ChatPage({
  localServicesOnly = false,
  inferenceUnavailable = false,
}: {
  localServicesOnly?: boolean;
  inferenceUnavailable?: boolean;
}) {
  const systemPanelOpen = useAppStore((s) => s.systemPanelOpen);

  return (
    <div className="flex h-full overflow-hidden">
      <div className="flex-1 min-w-0">
        <ChatArea localServicesOnly={localServicesOnly} inferenceUnavailable={inferenceUnavailable} />
      </div>
      {systemPanelOpen && <SystemPanel />}
    </div>
  );
}
