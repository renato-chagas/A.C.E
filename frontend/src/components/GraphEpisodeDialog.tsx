import { useEffect, useState, type RefObject } from 'react';
import { fetchMemoryEpisode, type MemoryEpisode } from '../lib/api';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from './ui/dialog';

export function GraphEpisodeTranscript({ episode }: { episode: MemoryEpisode }) {
  const date = new Date(episode.created_at * 1000);
  return (
    <div className="min-w-0 space-y-4">
      <div className="space-y-1 text-xs" style={{ color: 'var(--color-text-secondary)' }}>
        {!Number.isNaN(date.getTime()) && <time dateTime={date.toISOString()}>{date.toLocaleString()}</time>}
        {episode.source && <p className="break-all">Source: {episode.source}</p>}
      </div>
      <section className="space-y-1">
        <h3 className="font-medium">User</h3>
        <p className="whitespace-pre-wrap [overflow-wrap:anywhere]">{episode.user_text}</p>
      </section>
      <section className="space-y-1">
        <h3 className="font-medium">A.C.E.</h3>
        <p className="whitespace-pre-wrap [overflow-wrap:anywhere]">{episode.assistant_text}</p>
      </section>
    </div>
  );
}

function EpisodeDetails({ episodeId }: { episodeId: string }) {
  const [episode, setEpisode] = useState<MemoryEpisode | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    fetchMemoryEpisode(episodeId, controller.signal).then(
      (record) => { if (!controller.signal.aborted) setEpisode(record); },
      (reason) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Could not load this conversation episode.');
        }
      },
    );
    return () => controller.abort();
  }, [episodeId]);

  if (error) return <p role="alert" style={{ color: 'var(--color-error)' }}>{error}</p>;
  if (!episode) return <p role="status">Loading conversation episode…</p>;
  return <GraphEpisodeTranscript episode={episode} />;
}

export function GraphEpisodeDialog({ episodeId, onClose, returnFocus }: {
  episodeId: string | null;
  onClose: () => void;
  returnFocus: RefObject<HTMLButtonElement | null>;
}) {
  return (
    <Dialog open={episodeId !== null} onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent
        className="max-h-[85dvh] overflow-y-auto sm:max-w-2xl"
        finalFocus={returnFocus}
        style={{ background: 'var(--color-bg)', color: 'var(--color-text)' }}
      >
        <DialogHeader className="pr-8">
          <DialogTitle>Conversation episode</DialogTitle>
          <DialogDescription className="break-all"><code>{episodeId}</code></DialogDescription>
        </DialogHeader>
        {episodeId && <EpisodeDetails key={episodeId} episodeId={episodeId} />}
      </DialogContent>
    </Dialog>
  );
}
