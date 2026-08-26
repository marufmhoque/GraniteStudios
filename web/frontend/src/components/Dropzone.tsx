import { useCallback, useRef, useState } from 'react';
import { Upload } from 'lucide-react';

export function Dropzone({
  onFiles,
  busy,
  maxMb,
}: {
  onFiles: (files: File[]) => void;
  busy: boolean;
  maxMb: number;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragging(false);
      const files = Array.from(event.dataTransfer.files);
      if (files.length) onFiles(files);
    },
    [onFiles],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !busy && inputRef.current?.click()}
      className={`border-2 border-dashed rounded-2xl px-6 py-10 text-center cursor-pointer transition-colors ${
        dragging ? 'border-brand-gold bg-brand-gold/5' : 'border-brand-teal/25 hover:border-brand-teal/50'
      } ${busy ? 'opacity-50 pointer-events-none' : ''}`}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept="image/*,video/*,.nef,.cr2,.cr3,.arw,.dng,.raf,.orf,.rw2,.heic,.heif"
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length) onFiles(files);
          e.target.value = '';
        }}
      />
      <Upload className="w-7 h-7 mx-auto text-brand-teal/50" strokeWidth={1.5} />
      <p className="mt-3 font-serif text-lg text-brand-teal">Drop photos or video here</p>
      <p className="text-sm text-brand-teal/60 mt-1">
        JPEG, PNG, HEIC, RAW (NEF/CR2/ARW/DNG) &middot; MP4, MOV, AVI, MKV, WebM &middot; up to {maxMb}MB
      </p>
    </div>
  );
}
