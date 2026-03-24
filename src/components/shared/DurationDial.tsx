// Copyright (c) 2026 Ember Contributors. MIT License.
import { Component, Show, createSignal, createEffect, onCleanup } from 'solid-js';

interface DurationDialProps {
  value: number;
  min?: number;
  max?: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}

// Logarithmic dial for simulation duration (ns)
// Arc spans 270° (from 135° to 405°, i.e. gap at bottom-left)
const ARC_START = 135;
const ARC_SWEEP = 270;
const RADIUS = 56;
const CX = 70;
const CY = 70;
const TRACK_WIDTH = 8;

// Build snap values from explicit step rules.
// Keep fine control in the 1-10 us decade where users make longer-production adjustments.
const SNAP_VALUES: number[] = [];
{
  const ranges: [number, number, number][] = [
    // [start, end, step]
    [0.1, 1.0, 0.1],   // 0.1, 0.2, ... 1.0
    [1.2, 3.0, 0.2],   // 1.2, 1.4, ... 3.0
    [3.5, 10, 0.5],    // 3.5, 4.0, ... 10
    [11, 25, 1],        // 11, 12, ... 25
    [30, 100, 5],       // 30, 35, ... 100
    [125, 250, 25],     // 125, 150, ... 250
    [300, 500, 50],     // 300, 350, ... 500
    [600, 1000, 100],   // 600, 700, ... 1000
    [1100, 10000, 100], // 1.1, 1.2, ... 10.0 us
  ];
  for (const [start, end, step] of ranges) {
    for (let v = start; v <= end + step * 0.01; v += step) {
      SNAP_VALUES.push(Math.round(v * 100) / 100);
    }
  }
}

const MAJOR_TICK_LABELS = [0.1, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000];
const MINOR_TICKS = [2000, 3000, 4000, 6000, 7000, 8000, 9000];

const toLog = (v: number, min: number, max: number) =>
  (Math.log10(v) - Math.log10(min)) / (Math.log10(max) - Math.log10(min));

const fromLog = (t: number, min: number, max: number) =>
  Math.pow(10, Math.log10(min) + t * (Math.log10(max) - Math.log10(min)));

const polarToXY = (angleDeg: number, r: number) => ({
  x: CX + r * Math.cos((angleDeg * Math.PI) / 180),
  y: CY + r * Math.sin((angleDeg * Math.PI) / 180),
});

const describeArc = (startAngle: number, endAngle: number, r: number) => {
  const start = polarToXY(startAngle, r);
  const end = polarToXY(endAngle, r);
  const largeArc = endAngle - startAngle > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`;
};

const snapToNearest = (raw: number): number => {
  let best = SNAP_VALUES[0];
  let bestDist = Infinity;
  for (const v of SNAP_VALUES) {
    const dist = Math.abs(Math.log10(raw) - Math.log10(v));
    if (dist < bestDist) {
      bestDist = dist;
      best = v;
    }
  }
  return best;
};

const formatValue = (v: number): string => {
  if (v >= 1000) {
    const us = v / 1000;
    return us === Math.round(us) ? us.toFixed(0) : us.toFixed(1);
  }
  if (v >= 1) return v === Math.round(v) ? v.toFixed(0) : v.toFixed(1);
  return v.toFixed(1);
};

const formatUnit = (v: number): string => {
  if (v >= 1000) return 'us';
  return 'ns';
};

const formatTickLabel = (v: number): string => {
  if (v >= 1000) return `${v / 1000}us`;
  return `${v}`;
};

const DurationDial: Component<DurationDialProps> = (props) => {
  const min = () => props.min ?? 0.1;
  const max = () => props.max ?? 10000;

  const [isDragging, setIsDragging] = createSignal(false);
  const [isEditing, setIsEditing] = createSignal(false);
  const [editText, setEditText] = createSignal('');
  let svgRef: SVGSVGElement | undefined;
  let inputRef: HTMLInputElement | undefined;

  const valueToAngle = (v: number) => ARC_START + toLog(v, min(), max()) * ARC_SWEEP;
  const angleToValue = (angle: number) => {
    let t = (angle - ARC_START) / ARC_SWEEP;
    t = Math.max(0, Math.min(1, t));
    return fromLog(t, min(), max());
  };

  const getAngleFromEvent = (e: MouseEvent | TouchEvent): number => {
    if (!svgRef) return ARC_START;
    const rect = svgRef.getBoundingClientRect();
    const scaleX = 140 / rect.width;
    const scaleY = 140 / rect.height;
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    const mx = (clientX - rect.left) * scaleX;
    const my = (clientY - rect.top) * scaleY;
    let angle = (Math.atan2(my - CY, mx - CX) * 180) / Math.PI;
    if (angle < 0) angle += 360;
    // Handle the gap: if angle is in the dead zone, clamp to nearest end
    if (angle < ARC_START && angle < (ARC_START + ARC_SWEEP) % 360) {
      const distToStart = Math.abs(angle + 360 - ARC_START);
      const distToEnd = Math.abs(angle - ((ARC_START + ARC_SWEEP) % 360));
      angle = distToStart < distToEnd ? ARC_START : ARC_START + ARC_SWEEP;
    }
    return angle;
  };

  const handlePointerDown = (e: MouseEvent) => {
    if (props.disabled || isEditing()) return;
    e.preventDefault();
    setIsDragging(true);
    const angle = getAngleFromEvent(e);
    const raw = angleToValue(angle);
    props.onChange(snapToNearest(raw));
  };

  const handlePointerMove = (e: MouseEvent) => {
    if (!isDragging()) return;
    e.preventDefault();
    const angle = getAngleFromEvent(e);
    const raw = angleToValue(angle);
    props.onChange(snapToNearest(raw));
  };

  const handlePointerUp = () => {
    setIsDragging(false);
  };

  createEffect(() => {
    if (isDragging()) {
      document.addEventListener('mousemove', handlePointerMove);
      document.addEventListener('mouseup', handlePointerUp);
    }
    onCleanup(() => {
      document.removeEventListener('mousemove', handlePointerMove);
      document.removeEventListener('mouseup', handlePointerUp);
    });
  });

  // Click center value to enter edit mode
  const handleCenterClick = (e: MouseEvent) => {
    if (props.disabled) return;
    e.stopPropagation();
    e.preventDefault();
    setEditText(String(props.value));
    setIsEditing(true);
    requestAnimationFrame(() => {
      inputRef?.focus();
      inputRef?.select();
    });
  };

  const commitEdit = () => {
    const parsed = parseFloat(editText());
    if (!isNaN(parsed) && parsed > 0) {
      // Clamp to valid range but allow any number (no snap)
      const clamped = Math.max(0.1, Math.min(10000, parsed));
      props.onChange(clamped);
    }
    setIsEditing(false);
  };

  const cancelEdit = () => {
    setIsEditing(false);
  };

  const handleEditKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      commitEdit();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      cancelEdit();
    }
  };

  const thumbAngle = () => valueToAngle(Math.max(min(), Math.min(max(), props.value)));
  const thumb = () => polarToXY(thumbAngle(), RADIUS);

  // Filled arc from start to current value
  const filledArcPath = () => {
    const endAngle = thumbAngle();
    if (endAngle <= ARC_START + 0.5) return '';
    return describeArc(ARC_START, endAngle, RADIUS);
  };

  return (
    <div class="flex flex-col items-center select-none relative">
      <svg
        ref={svgRef}
        viewBox="-15 -15 170 170"
        width="160"
        height="160"
        overflow="visible"
        class={`${props.disabled ? 'opacity-50' : 'cursor-pointer'}`}
        onMouseDown={handlePointerDown}
      >
        {/* Track background */}
        <path
          d={describeArc(ARC_START, ARC_START + ARC_SWEEP, RADIUS)}
          fill="none"
          stroke="oklch(var(--b3))"
          stroke-width={TRACK_WIDTH}
          stroke-linecap="round"
        />

        {/* Filled arc */}
        <path
          d={filledArcPath()}
          fill="none"
          stroke="oklch(var(--p))"
          stroke-width={TRACK_WIDTH}
          stroke-linecap="round"
        />

        {/* Minor ticks in the 1-10 us decade for smoother visual guidance */}
        {MINOR_TICKS.map((v) => {
          const angle = valueToAngle(v);
          const inner = polarToXY(angle, RADIUS - TRACK_WIDTH / 2 - 1);
          const outer = polarToXY(angle, RADIUS + TRACK_WIDTH / 2 + 1);
          return (
            <line
              x1={inner.x} y1={inner.y}
              x2={outer.x} y2={outer.y}
              stroke="oklch(var(--bc) / 0.22)"
              stroke-width="1"
            />
          );
        })}

        {/* Major tick marks and labels */}
        {MAJOR_TICK_LABELS.map((v) => {
          const angle = valueToAngle(v);
          const inner = polarToXY(angle, RADIUS - TRACK_WIDTH / 2 - 2);
          const outer = polarToXY(angle, RADIUS + TRACK_WIDTH / 2 + 2);
          const labelPos = polarToXY(angle, RADIUS + TRACK_WIDTH / 2 + 13);
          return (
            <>
              <line
                x1={inner.x} y1={inner.y}
                x2={outer.x} y2={outer.y}
                stroke="oklch(var(--bc) / 0.3)"
                stroke-width="1"
              />
              <text
                x={labelPos.x}
                y={labelPos.y}
                text-anchor="middle"
                dominant-baseline="central"
                fill="oklch(var(--bc) / 0.5)"
                font-size="8"
                font-family="monospace"
              >
                {formatTickLabel(v)}
              </text>
            </>
          );
        })}

        {/* Thumb */}
        <circle
          cx={thumb().x}
          cy={thumb().y}
          r={TRACK_WIDTH / 2 + 3}
          fill="oklch(var(--p))"
          stroke="oklch(var(--b1))"
          stroke-width="2"
          class={isDragging() ? '' : 'transition-all duration-100'}
        />

        {/* Center value display (clickable) — hidden when editing */}
        <Show when={!isEditing()}>
          <g
            style={{ cursor: props.disabled ? 'default' : 'text' }}
            onMouseDown={(e) => { e.stopPropagation(); handleCenterClick(e); }}
          >
            {/* Invisible hit area */}
            <rect
              x={CX - 30} y={CY - 18}
              width="60" height="36"
              fill="transparent"
            />
            <text
              x={CX}
              y={CY - 6}
              text-anchor="middle"
              dominant-baseline="central"
              fill="oklch(var(--bc))"
              font-size="22"
              font-weight="bold"
              font-family="monospace"
            >
              {formatValue(props.value)}
            </text>
            <text
              x={CX}
              y={CY + 14}
              text-anchor="middle"
              dominant-baseline="central"
              fill="oklch(var(--bc) / 0.6)"
              font-size="12"
              font-family="monospace"
            >
              {formatUnit(props.value)}
            </text>
          </g>
        </Show>
      </svg>

      {/* Inline edit input — overlaid on center when editing */}
      <Show when={isEditing()}>
        <div
          class="absolute flex flex-col items-center"
          style={{ top: '50%', left: '50%', transform: 'translate(-50%, -60%)' }}
        >
          <input
            ref={inputRef}
            type="number"
            class="input input-bordered input-sm w-20 text-center font-mono text-lg font-bold p-0 h-8"
            value={editText()}
            onInput={(e) => setEditText(e.currentTarget.value)}
            onKeyDown={handleEditKeyDown}
            onBlur={commitEdit}
            min="0.1"
            max="10000"
            step="any"
          />
          <span class="text-[10px] text-base-content/50 mt-0.5">ns (Enter to confirm)</span>
        </div>
      </Show>
    </div>
  );
};

export default DurationDial;
