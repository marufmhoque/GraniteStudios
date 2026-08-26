import { useState } from 'react';
import { Plus, Trash2, Play, ArrowDown } from 'lucide-react';
import type { Capabilities, EditResult, UploadedFile, VideoOperationSpec } from '../api';
import { startVideoEdit, waitForJob } from '../api';
import { Button, Field, Notice, Panel, Select, Spinner, humanize } from './ui';

interface Props {
  capabilities: Capabilities;
  selected: UploadedFile[];
  onResult: (result: EditResult) => void;
}

// Parameter schema per operation, so the builder renders the right inputs
// and sends defaults the backend engines actually accept.
const OP_PARAMS: Record<string, { key: string; label: string; default: number | string; step?: number; options?: string[] }[]> = {
  trim: [
    { key: 'start_sec', label: 'Start (s)', default: 0, step: 0.5 },
    { key: 'end_sec', label: 'End (s)', default: 5, step: 0.5 },
  ],
  reverse: [],
  change_speed: [{ key: 'factor', label: 'Speed ×', default: 2, step: 0.25 }],
  timelapse: [{ key: 'target_duration_sec', label: 'Target length (s)', default: 30, step: 1 }],
  freeze_frame: [
    { key: 'at_sec', label: 'Freeze at (s)', default: 1, step: 0.5 },
    { key: 'hold_sec', label: 'Hold (s)', default: 2, step: 0.5 },
  ],
  convert_aspect_ratio: [
    { key: 'target', label: 'Aspect', default: '9:16', options: [] },
    { key: 'mode', label: 'Mode', default: 'crop', options: ['crop', 'pad'] },
  ],
  auto_crop_subject: [{ key: 'target_aspect', label: 'Aspect', default: '9:16', options: [] }],
  stabilize: [{ key: 'smoothing_window', label: 'Smoothing frames', default: 30, step: 5 }],
  color_correct: [
    { key: 'brightness', label: 'Brightness', default: 0, step: 0.05 },
    { key: 'contrast', label: 'Contrast', default: 1, step: 0.05 },
    { key: 'saturation', label: 'Saturation', default: 1, step: 0.05 },
  ],
  reduce_noise: [{ key: 'strength', label: 'Strength (dB)', default: 25, step: 5 }],
  normalize_loudness: [{ key: 'target_lufs', label: 'Target LUFS', default: -16, step: 1 }],
};

interface Step {
  id: number;
  name: string;
  params: Record<string, number | string>;
}

let nextStepId = 1;

export function VideoPanel({ capabilities, selected, onResult }: Props) {
  const [steps, setSteps] = useState<Step[]>([]);
  const [pendingOp, setPendingOp] = useState(capabilities.video.operations[0]);
  const [preset, setPreset] = useState('web_mp4');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  const target = selected.find((f) => f.kind === 'video');

  function addStep() {
    const spec = OP_PARAMS[pendingOp] ?? [];
    const params: Record<string, number | string> = {};
    for (const field of spec) {
      params[field.key] =
        field.options && field.options.length === 0
          ? capabilities.video.aspect_ratios[0] ?? '9:16'
          : field.default;
    }
    setSteps((prev) => [...prev, { id: nextStepId++, name: pendingOp, params }]);
  }

  function updateParam(stepId: number, key: string, value: string) {
    setSteps((prev) =>
      prev.map((step) =>
        step.id === stepId
          ? {
              ...step,
              params: {
                ...step.params,
                // Numeric fields keep numbers; aspect/mode selects stay strings.
                [key]: /^-?\d*\.?\d*$/.test(value) && value !== '' ? Number(value) : value,
              },
            }
          : step,
      ),
    );
  }

  async function run() {
    if (!target) return;
    setBusy('Queueing job');
    setError('');
    try {
      const operations: VideoOperationSpec[] = steps.map((s) => ({ name: s.name, params: s.params }));
      const { job_id } = await startVideoEdit(target.file_id, operations, preset);
      const status = await waitForJob(job_id, (s) => setBusy(s.message || 'Processing'));
      if (status.status === 'failed' || !status.result_file_id) {
        throw new Error(status.error ?? 'Video processing failed');
      }
      onResult({
        file_id: status.result_file_id,
        name: `${target.name.replace(/\.[^.]+$/, '')}_edited`,
        kind: 'video',
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy('');
    }
  }

  if (!target) {
    return (
      <Panel title="Video Editor" subtitle="Select a video above to build an edit.">
        <p className="text-sm text-brand-teal/60">
          Chain trims, speed ramps, reframing, stabilization and color correction into a single
          render — the same FFmpeg + OpenCV engine as the desktop app.
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
        title="Edit Chain"
        subtitle={`Operations run top to bottom on ${target.name}${
          target.video ? ` — ${target.video.width}×${target.video.height}, ${target.video.duration_label}` : ''
        }.`}
      >
        <div className="flex flex-wrap gap-3 items-end">
          <div className="flex-1 min-w-[200px]">
            <Field label="Add operation">
              <Select
                value={pendingOp}
                onChange={setPendingOp}
                options={capabilities.video.operations.map((op) => ({
                  value: op,
                  label: humanize(op),
                }))}
              />
            </Field>
          </div>
          <Button variant="secondary" onClick={addStep} disabled={!!busy}>
            <span className="inline-flex items-center gap-2">
              <Plus className="w-4 h-4" /> Add Step
            </span>
          </Button>
        </div>

        {steps.length === 0 ? (
          <p className="text-sm text-brand-teal/50 mt-5">
            No steps yet — add one above, or export as-is to just transcode to a new preset.
          </p>
        ) : (
          <ol className="mt-5 space-y-3">
            {steps.map((step, index) => (
              <li key={step.id}>
                {index > 0 && (
                  <ArrowDown className="w-4 h-4 text-brand-teal/30 mx-auto mb-3" strokeWidth={1.5} />
                )}
                <div className="border border-brand-teal/15 rounded-xl p-4 bg-white/60">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-medium text-sm text-brand-teal">
                      {index + 1}. {humanize(step.name)}
                    </span>
                    <button
                      onClick={() => setSteps((prev) => prev.filter((s) => s.id !== step.id))}
                      className="text-brand-teal/40 hover:text-red-600 transition-colors"
                      aria-label={`Remove ${humanize(step.name)}`}
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>

                  {(OP_PARAMS[step.name] ?? []).length > 0 && (
                    <div className="grid sm:grid-cols-3 gap-3 mt-3">
                      {(OP_PARAMS[step.name] ?? []).map((field) => {
                        const choices =
                          field.options && field.options.length === 0
                            ? capabilities.video.aspect_ratios
                            : field.options;
                        return (
                          <Field key={field.key} label={field.label}>
                            {choices ? (
                              <Select
                                value={String(step.params[field.key])}
                                onChange={(v) => updateParam(step.id, field.key, v)}
                                options={choices.map((c) => ({ value: c, label: c }))}
                              />
                            ) : (
                              <input
                                type="number"
                                step={field.step}
                                value={String(step.params[field.key])}
                                onChange={(e) => updateParam(step.id, field.key, e.target.value)}
                                className="w-full px-3 py-2 rounded-lg border border-brand-teal/20 bg-white text-sm text-brand-teal focus:outline-none focus:border-brand-gold"
                              />
                            )}
                          </Field>
                        );
                      })}
                    </div>
                  )}
                </div>
              </li>
            ))}
          </ol>
        )}
      </Panel>

      <Panel title="Export">
        <div className="grid sm:grid-cols-[1fr_auto] gap-3 items-end">
          <Field label="Preset">
            <Select
              value={preset}
              onChange={setPreset}
              options={Object.entries(capabilities.video.presets).map(([value, label]) => ({
                value,
                label,
              }))}
            />
          </Field>
          <Button onClick={run} disabled={!!busy}>
            <span className="inline-flex items-center gap-2">
              <Play className="w-4 h-4" /> Render Video
            </span>
          </Button>
        </div>
        <p className="text-xs text-brand-teal/50 mt-3">
          Stabilization and long clips take the longest — rendering runs on the server and this page
          polls until it finishes.
        </p>
      </Panel>
    </div>
  );
}
