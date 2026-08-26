// Typed client for the Granite Studios web API.
//
// In dev, requests go to /api and Vite proxies them to the local backend.
// In production VITE_API_URL points at the deployed Render service.

const API_BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '');

export interface VideoInfo {
  width: number;
  height: number;
  fps: number;
  duration_sec: number;
  duration_label: string;
  has_audio: boolean;
  codec: string;
}

export interface UploadedFile {
  file_id: string;
  name: string;
  kind: 'image' | 'video';
  size_mb: number;
  video?: VideoInfo;
  metadata?: { gear: string; settings: string };
}

export interface EditResult {
  file_id: string;
  name: string;
  kind: 'image' | 'video';
}

export interface JobStatus {
  job_id: string;
  kind: string;
  status: 'queued' | 'running' | 'done' | 'failed';
  message: string;
  error: string | null;
  result_file_id: string | null;
}

export interface Capabilities {
  photo: { filters: string[]; colors: string[]; operations: string[] };
  video: {
    operations: string[];
    aspect_ratios: string[];
    presets: Record<string, string>;
    transitions: string[];
  };
  limits: { max_upload_mb: number };
}

export interface VideoOperationSpec {
  name: string;
  params: Record<string, unknown>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      // Non-JSON error body (e.g. a proxy timeout page) — keep the status text.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const fileUrl = (fileId: string, download = false) =>
  `${API_BASE}/api/file/${fileId}${download ? '?download=true' : ''}`;

export const previewUrl = (fileId: string) => `${API_BASE}/api/preview/${fileId}`;

export const getCapabilities = () => request<Capabilities>('/api/capabilities');

export const createSession = () =>
  request<{ session_id: string }>('/api/session', { method: 'POST' });

export async function uploadFile(sessionId: string, file: File): Promise<UploadedFile> {
  const form = new FormData();
  form.append('session_id', sessionId);
  form.append('file', file);
  return request<UploadedFile>('/api/upload', { method: 'POST', body: form });
}

const jsonPost = <T,>(path: string, payload: unknown) =>
  request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

export const editPhoto = (payload: {
  file_id: string;
  operation: string;
  filter_name?: string;
  color_name?: string;
  brightness?: number;
  contrast?: number;
  saturation?: number;
}) => jsonPost<EditResult>('/api/photo/edit', payload);

export const formatForInstagram = (fileId: string, target: 'feed' | 'story') =>
  jsonPost<EditResult>('/api/photo/instagram', { file_id: fileId, target });

export const startVideoEdit = (
  fileId: string,
  operations: VideoOperationSpec[],
  preset: string,
) => jsonPost<{ job_id: string }>('/api/video/edit', { file_id: fileId, operations, preset });

export const startReel = (fileIds: string[], duration: number) =>
  jsonPost<{ job_id: string }>('/api/video/reel', { file_ids: fileIds, duration });

export const getJob = (jobId: string) => request<JobStatus>(`/api/job/${jobId}`);

/** Poll a job to completion, reporting progress messages as they change. */
export async function waitForJob(
  jobId: string,
  onProgress: (status: JobStatus) => void,
  intervalMs = 1500,
): Promise<JobStatus> {
  for (;;) {
    const status = await getJob(jobId);
    onProgress(status);
    if (status.status === 'done' || status.status === 'failed') return status;
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}
