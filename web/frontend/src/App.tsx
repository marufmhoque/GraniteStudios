import { useEffect, useState } from 'react';
import { Download, Image as ImageIcon, Video, X, ExternalLink } from 'lucide-react';
import type { Capabilities, EditResult, UploadedFile } from './api';
import { createSession, fileUrl, getCapabilities, previewUrl, uploadFile } from './api';
import { Dropzone } from './components/Dropzone';
import { PhotoPanel } from './components/PhotoPanel';
import { VideoPanel } from './components/VideoPanel';
import { Notice, Spinner } from './components/ui';

type Tab = 'photo' | 'video';

export default function App() {
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [sessionId, setSessionId] = useState('');
  const [bootError, setBootError] = useState('');
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [results, setResults] = useState<EditResult[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [tab, setTab] = useState<Tab>('photo');

  useEffect(() => {
    (async () => {
      try {
        const [caps, session] = await Promise.all([getCapabilities(), createSession()]);
        setCapabilities(caps);
        setSessionId(session.session_id);
      } catch (err) {
        setBootError(
          err instanceof Error
            ? `${err.message}. The processing server may be waking up — this can take up to a minute on first load.`
            : String(err),
        );
      }
    })();
  }, []);

  async function handleFiles(incoming: File[]) {
    setUploading(true);
    setUploadError('');
    try {
      for (const file of incoming) {
        const uploaded = await uploadFile(sessionId, file);
        setFiles((prev) => [...prev, uploaded]);
        setSelectedIds((prev) => [...prev, uploaded.file_id]);
        if (uploaded.kind === 'video') setTab('video');
      }
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  }

  const toggleSelect = (fileId: string) =>
    setSelectedIds((prev) =>
      prev.includes(fileId) ? prev.filter((id) => id !== fileId) : [...prev, fileId],
    );

  const removeFile = (fileId: string) => {
    setFiles((prev) => prev.filter((f) => f.file_id !== fileId));
    setSelectedIds((prev) => prev.filter((id) => id !== fileId));
  };

  const selected = files.filter((f) => selectedIds.includes(f.file_id));

  if (bootError) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="max-w-md">
          <Notice kind="error">{bootError}</Notice>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 text-sm text-brand-teal underline"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!capabilities || !sessionId) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Spinner label="Starting Granite Studios…" />
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-brand-teal/10 bg-brand-ivory/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-5 sm:px-8 py-5 flex items-center justify-between gap-4">
          <div>
            <h1 className="font-serif text-2xl sm:text-3xl text-brand-teal">Granite Studios</h1>
            <p className="text-sm text-brand-teal/60">AI photo &amp; video suite, running in your browser</p>
          </div>
          <a
            href="https://github.com/marufmhoque/GraniteStudios"
            target="_blank"
            rel="noreferrer"
            className="hidden sm:inline-flex items-center gap-1.5 text-sm text-brand-teal/70 hover:text-brand-teal"
          >
            Desktop app <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-5 sm:px-8 py-8 space-y-8">
        <Dropzone onFiles={handleFiles} busy={uploading} maxMb={capabilities.limits.max_upload_mb} />
        {uploading && <Spinner label="Uploading…" />}
        {uploadError && <Notice kind="error">{uploadError}</Notice>}

        {files.length > 0 && (
          <section>
            <h2 className="font-serif text-xl text-brand-teal mb-3">
              Your Files
              <span className="text-sm font-sans text-brand-teal/50 ml-2">
                {selected.length} of {files.length} selected
              </span>
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
              {files.map((file) => {
                const isSelected = selectedIds.includes(file.file_id);
                return (
                  <div
                    key={file.file_id}
                    onClick={() => toggleSelect(file.file_id)}
                    className={`relative rounded-xl overflow-hidden border-2 cursor-pointer transition-all bg-white ${
                      isSelected ? 'border-brand-gold' : 'border-transparent hover:border-brand-teal/20'
                    }`}
                  >
                    <div className="aspect-square bg-brand-teal/5 flex items-center justify-center">
                      {file.kind === 'image' ? (
                        <img
                          src={previewUrl(file.file_id)}
                          alt={file.name}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <Video className="w-8 h-8 text-brand-teal/40" strokeWidth={1.5} />
                      )}
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeFile(file.file_id);
                      }}
                      className="absolute top-2 right-2 bg-white/90 rounded-full p-1 text-brand-teal/60 hover:text-red-600"
                      aria-label={`Remove ${file.name}`}
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                    <div className="p-2">
                      <p className="text-xs text-brand-teal truncate" title={file.name}>
                        {file.name}
                      </p>
                      <p className="text-[11px] text-brand-teal/50">
                        {file.video
                          ? `${file.video.width}×${file.video.height} · ${file.video.duration_label}`
                          : `${file.size_mb} MB`}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        <div className="flex gap-2 border-b border-brand-teal/10">
          {(['photo', 'video'] as Tab[]).map((name) => (
            <button
              key={name}
              onClick={() => setTab(name)}
              className={`px-4 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors inline-flex items-center gap-2 ${
                tab === name
                  ? 'border-brand-gold text-brand-teal'
                  : 'border-transparent text-brand-teal/50 hover:text-brand-teal'
              }`}
            >
              {name === 'photo' ? <ImageIcon className="w-4 h-4" /> : <Video className="w-4 h-4" />}
              {name === 'photo' ? 'Photo Tools' : 'Video Editor'}
            </button>
          ))}
        </div>

        {tab === 'photo' ? (
          <PhotoPanel
            capabilities={capabilities}
            selected={selected}
            onResult={(r) => setResults((prev) => [r, ...prev])}
          />
        ) : (
          <VideoPanel
            capabilities={capabilities}
            selected={selected}
            onResult={(r) => setResults((prev) => [r, ...prev])}
          />
        )}

        {results.length > 0 && (
          <section>
            <h2 className="font-serif text-xl text-brand-teal mb-3">Results</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
              {results.map((result, index) => (
                <div
                  key={`${result.file_id}-${index}`}
                  className="rounded-xl overflow-hidden border border-brand-teal/15 bg-white"
                >
                  <div className="aspect-square checkerboard flex items-center justify-center">
                    {result.kind === 'image' ? (
                      <img
                        src={fileUrl(result.file_id)}
                        alt={result.name}
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <video src={fileUrl(result.file_id)} controls className="w-full h-full object-contain" />
                    )}
                  </div>
                  <div className="p-3 flex items-center justify-between gap-2">
                    <p className="text-xs text-brand-teal truncate" title={result.name}>
                      {result.name}
                    </p>
                    <a
                      href={fileUrl(result.file_id, true)}
                      download
                      className="shrink-0 text-brand-teal/60 hover:text-brand-gold"
                      aria-label={`Download ${result.name}`}
                    >
                      <Download className="w-4 h-4" />
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        <footer className="pt-6 border-t border-brand-teal/10 text-xs text-brand-teal/50">
          Uploads are processed on demand and deleted automatically within an hour. Nothing is stored
          permanently or shared.
        </footer>
      </main>
    </div>
  );
}
