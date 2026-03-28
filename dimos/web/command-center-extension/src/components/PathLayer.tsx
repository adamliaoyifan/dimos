import * as d3 from "d3";
import * as React from "react";

import { Path } from "../types";

interface PathLayerProps {
  path: Path;
  worldToPx: (x: number, y: number) => [number, number];
  color?: string;
  label?: string;
}

const PathLayer = React.memo<PathLayerProps>(({ path, worldToPx, color = "#ff3333", label }) => {
  const points = React.useMemo(
    () => path.coords.map(([x, y]) => worldToPx(x, y)),
    [path.coords, worldToPx],
  );

  const pathData = React.useMemo(() => {
    const line = d3.line();
    return line(points);
  }, [points]);

  const gradientId = React.useMemo(() => `path-gradient-${color.replace("#", "")}-${Date.now()}`, []);

  if (path.coords.length < 2) {
    return null;
  }

  const lastPoint = points[points.length - 1]!;

  return (
    <>
      <defs>
        <linearGradient
          id={gradientId}
          gradientUnits="userSpaceOnUse"
          x1={points[0]![0]}
          y1={points[0]![1]}
          x2={lastPoint[0]}
          y2={lastPoint[1]}
        >
          <stop offset="0%" stopColor={color} />
          <stop offset="100%" stopColor={color} />
        </linearGradient>
      </defs>
      <path
        d={pathData ?? ""}
        fill="none"
        stroke={`url(#${gradientId})`}
        strokeWidth={5}
        strokeLinecap="round"
        opacity={0.9}
      />
      {label && (
        <text
          x={lastPoint[0]}
          y={lastPoint[1] - 10}
          fill={color}
          fontSize={11}
          fontWeight="bold"
          textAnchor="middle"
          style={{ pointerEvents: "none", textShadow: "0 0 3px #000" }}
        >
          {label}
        </text>
      )}
    </>
  );
});

PathLayer.displayName = "PathLayer";

export default PathLayer;
