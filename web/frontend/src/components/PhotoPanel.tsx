import { useState } from 'react';
import { Sparkles, Scissors, Palette, Wand2, Instagram, Film } from 'lucide-react';
import type { Capabilities, EditResult, UploadedFile } from '../api';
import { editPhoto, formatForInstagram, startReel, waitForJob } from '../api';
import { Button, Field, Notice, Panel, Select, Slider, Spinner, humanize } from './ui';

interface Props {
  capabilities: Capabilities;
  selected: UploadedFile[];
  onResult: (result: EditResult) => void;
}

export function PhotoPanel({ capabilities, selected, onResult }: Props) {
  const [filter, setFilter] = useState(capabilities.photo.filters[0]);
  const [color, setColor] = useState(capabilities.photo.colors[0]);
  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(1);
  const [saturation, setSaturation] = useState(1);
  const [reelDuration, setReelDuration] = useState(10);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  const targets = selected.filter((f) => f.kind === 'image');
  const hasTargets = targets.length > 0;

  async function runOnEach(label: string, run: (fileId: string) => Promise<EditResult>) {
    setBusy(label);
    setError('');
    try {
      for (const target of targets) {
        onResult(await run(target.file_id));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy('');
    }
  }

  async function makeReel() {
    setBusy('Rendering panoramic reel');
    setError('');
    try {
      const { job_id } = await startReel(targets.map((t) => t.file_id), reelDuration);
      const status = await waitForJob(job_id, (s) => setBusy(s.message || 'Rendering reel'));
      if (status.status === 'failed' || !status.result_file_id) {
        throw new Error(status.error ?? 'Reel rendering failed');
      }
      onResult({ file_id: status.result_file_id, name: 'panoramic_reel.mp4', kind: 'video' });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy('');
    }
  }

  if (!hasTargets) {
    return (
      <Panel title="Photo Tools" subtitle="Select one or more photos above to start editing.">
        <p className="text-sm text-brand-teal/60">
          Every edit runs the same engine as the Granite Studios desktop app — your originals are
          never modified, and each result is downloadable.
        </p>
      </Panel>
    );
  }

  return (
    <div className="space-y-5">
      {error && <Notice kind="error">{error}</Notice>}
      {busy && (
        <div className="bg-white/70 border border-brand-teal/10 rounded-xl px-4 py-3">
          <Spinner label={busy} />
        </div>
      )}

      <Panel
        title="One-Click AI"
        subtitle={`Applies to ${targets.length} selected photo${targets.length > 1 ? 's' : ''}.`}
      >
        <div className="flex flex-wrap gap-3">
          <Button
            disabled={!!busy}
            onClick={() =>
              runOnEach('Removing background', (file_id) =>
                editPhoto({ file_id, operation: 'remove_background' }),
              )
            }
          >
            <span className="inline-flex items-center gap-2">
              <Scissors className="w-4 h-4" /> Remove Background
            </span>
          </Button>
          <Button
            variant="secondary"
            disabled={!!busy}
            onClick={() =>
              runOnEach('Auto-fixing', (file_id) => editPhoto({ file_id, operation: 'auto_fix' }))
            }
          >
            <span className="inline-flex items-center gap-2">
              <Wand2 className="w-4 h-4" /> Smart Auto-Fix
            </span>
          </Button>
        </div>
        <p className="text-xs text-brand-teal/50 mt-3">
          Background removal outputs a transparent PNG. Auto-fix applies white-balance correction
          and adaptive histogram equalization.
        </p>
      </Panel>

      <Panel title="Cinematic Grading" subtitle="Matrix-based film grades with tone curves.">
        <div className="grid sm:grid-cols-[1fr_auto] gap-3 items-end">
          <Field label="Filter">
            <Select
              value={filter}
              onChange={setFilter}
              options={capabilities.photo.filters.map((f) => ({ value: f, label: humanize(f) }))}
            />
          </Field>
          <Button
            disabled={!!busy}
            onClick={() =>
              runOnEach(`Applying ${humanize(filter)}`, (file_id) =>
                editPhoto({ file_id, operation: 'cinematic_filter', filter_name: filter }),
              )
            }
          >
            <span className="inline-flex items-center gap-2">
              <Sparkles className="w-4 h-4" /> Apply Filter
            </span>
          </Button>
        </div>
      </Panel>

      <Panel title="Selective Color" subtitle="Keep one hue in color, render the rest in mono.">
        <div className="grid sm:grid-cols-[1fr_auto] gap-3 items-end">
          <Field label="Keep this color">
            <Select
              value={color}
              onChange={setColor}
              options={capabilities.photo.colors.map((c) => ({ value: c, label: humanize(c) }))}
            />
          </Field>
          <Button
            disabled={!!busy}
            onClick={() =>
              runOnEach(`Isolating ${color}`, (file_id) =>
                editPhoto({ file_id, operation: 'selective_color', color_name: color }),
              )
            }
          >
            <span className="inline-flex items-center gap-2">
              <Palette className="w-4 h-4" /> Isolate Color
            </span>
          </Button>
        </div>
      </Panel>

      <Panel title="Manual Adjustments">
        <div className="space-y-4">
          <Field label={`Brightness (${brightness})`}>
            <Slider value={brightness} onChange={setBrightness} min={-100} max={100} step={1} />
          </Field>
          <Field label={`Contrast (${contrast.toFixed(2)}×)`}>
            <Slider value={contrast} onChange={setContrast} min={0.2} max={2.5} step={0.05} />
          </Field>
          <Field label={`Saturation (${saturation.toFixed(2)}×)`}>
            <Slider value={saturation} onChange={setSaturation} min={0} max={3} step={0.05} />
          </Field>
          <Button
            disabled={!!busy}
            onClick={() =>
              runOnEach('Applying adjustments', (file_id) =>
                editPhoto({ file_id, operation: 'adjust', brightness, contrast, saturation }),
              )
            }
          >
            Apply Adjustments
          </Button>
        </div>
      </Panel>

      <Panel title="Social Exports" subtitle="Instagram-ready framing and panoramic reels.">
        <div className="flex flex-wrap gap-3">
          <Button
            variant="secondary"
            disabled={!!busy}
            onClick={() => runOnEach('Formatting for feed', (id) => formatForInstagram(id, 'feed'))}
          >
            <span className="inline-flex items-center gap-2">
              <Instagram className="w-4 h-4" /> Feed 4:5
            </span>
          </Button>
          <Button
            variant="secondary"
            disabled={!!busy}
            onClick={() => runOnEach('Formatting for story', (id) => formatForInstagram(id, 'story'))}
          >
            <span className="inline-flex items-center gap-2">
              <Instagram className="w-4 h-4" /> Story 9:16
            </span>
          </Button>
        </div>

        <div className="mt-5 pt-5 border-t border-brand-teal/10">
          <div className="grid sm:grid-cols-[1fr_auto] gap-3 items-end">
            <Field label={`Reel duration — ${reelDuration}s`}>
              <Slider value={reelDuration} onChange={setReelDuration} min={5} max={20} step={1} />
            </Field>
            <Button
              variant="secondary"
              disabled={!!busy || targets.length < 3 || targets.length > 7}
              onClick={makeReel}
            >
              <span className="inline-flex items-center gap-2">
                <Film className="w-4 h-4" /> Panoramic Reel
              </span>
            </Button>
          </div>
          <p className="text-xs text-brand-teal/50 mt-2">
            Select 3–7 photos to stitch one panorama and pan a 1080×1920 viewport across it
            (currently {targets.length} selected).
          </p>
        </div>
      </Panel>
    </div>
  );
}
