import type { ReactNode } from 'react';

export function Button({
  children,
  onClick,
  variant = 'primary',
  disabled,
  className = '',
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'ghost';
  disabled?: boolean;
  className?: string;
}) {
  const styles = {
    primary: 'bg-brand-teal text-brand-ivory hover:bg-brand-teal/90',
    secondary: 'bg-white text-brand-teal border border-brand-teal/20 hover:border-brand-teal/50',
    ghost: 'text-brand-teal/70 hover:text-brand-teal hover:bg-brand-teal/5',
  }[variant];

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${styles} ${className}`}
    >
      {children}
    </button>
  );
}

export function Panel({ title, subtitle, children }: { title: string; subtitle?: string; children: ReactNode }) {
  return (
    <section className="bg-white/70 border border-brand-teal/10 rounded-2xl p-5 sm:p-6">
      <h3 className="font-serif text-xl text-brand-teal">{title}</h3>
      {subtitle && <p className="text-sm text-brand-teal/60 mt-1 mb-4">{subtitle}</p>}
      <div className={subtitle ? '' : 'mt-4'}>{children}</div>
    </section>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="block text-xs uppercase tracking-wide text-brand-teal/60 mb-1.5">{label}</span>
      {children}
    </label>
  );
}

const controlClass =
  'w-full px-3 py-2 rounded-lg border border-brand-teal/20 bg-white text-sm text-brand-teal focus:outline-none focus:border-brand-gold';

export function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className={controlClass}>
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

export function NumberInput({
  value,
  onChange,
  step = 0.1,
  min,
  max,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  step?: number;
  min?: number;
  max?: number;
  placeholder?: string;
}) {
  return (
    <input
      type="number"
      value={value}
      step={step}
      min={min}
      max={max}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      className={controlClass}
    />
  );
}

export function Slider({
  value,
  onChange,
  min,
  max,
  step,
}: {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
}) {
  return (
    <div className="flex items-center gap-3">
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 accent-brand-gold"
      />
      <span className="text-xs tabular-nums text-brand-teal/70 w-12 text-right">{value.toFixed(2)}</span>
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-brand-teal/70">
      <span className="w-4 h-4 border-2 border-brand-teal/20 border-t-brand-gold rounded-full animate-spin" />
      {label}
    </div>
  );
}

export function Notice({ kind, children }: { kind: 'error' | 'info'; children: ReactNode }) {
  const styles =
    kind === 'error'
      ? 'bg-red-50 border-red-200 text-red-800'
      : 'bg-brand-gold/10 border-brand-gold/30 text-brand-teal';
  return <div className={`text-sm rounded-lg border px-4 py-3 ${styles}`}>{children}</div>;
}

/** Human-readable label for an engine identifier like "cinematic_teal_orange". */
export const humanize = (value: string) =>
  value.replace(/[_-]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
