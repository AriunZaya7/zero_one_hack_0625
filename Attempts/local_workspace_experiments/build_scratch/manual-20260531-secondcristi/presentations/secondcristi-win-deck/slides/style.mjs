export const C = {
  bg: "#F7F8FA",
  ink: "#111827",
  muted: "#5B6472",
  line: "#D9DEE7",
  panel: "#FFFFFF",
  blue: "#2563EB",
  blueSoft: "#DCEAFE",
  green: "#0F8A5F",
  greenSoft: "#DFF5EC",
  amber: "#B7791F",
  amberSoft: "#FFF3D6",
  red: "#B42318",
  redSoft: "#FFE4E0",
  dark: "#172033",
  cyan: "#0891B2",
};

export function bg(ctx, slide, fill = C.bg) {
  ctx.addShape(slide, { x: 0, y: 0, w: ctx.W, h: ctx.H, fill });
}

export function title(ctx, slide, text, sub = "") {
  ctx.addText(slide, {
    text,
    x: 60,
    y: 42,
    w: 900,
    h: 50,
    fontSize: 30,
    bold: true,
    color: C.ink,
    typeface: ctx.fonts.title,
  });
  if (sub) {
    ctx.addText(slide, {
      text: sub,
      x: 60,
      y: 90,
      w: 900,
      h: 32,
      fontSize: 15,
      color: C.muted,
    });
  }
  ctx.addShape(slide, { x: 60, y: 130, w: 1160, h: 1, fill: C.line });
}

export function foot(ctx, slide, text = "SECONDCRISTI | Industrial AI Infineon | May 31, 2026") {
  ctx.addText(slide, {
    text,
    x: 60,
    y: 680,
    w: 800,
    h: 20,
    fontSize: 10,
    color: "#7B8494",
  });
}

export function pill(ctx, slide, text, x, y, w, fill, color) {
  ctx.addShape(slide, {
    x,
    y,
    w,
    h: 28,
    fill,
    line: ctx.line(fill, 0),
  });
  ctx.addText(slide, {
    text,
    x: x + 12,
    y: y + 6,
    w: w - 24,
    h: 18,
    fontSize: 11,
    bold: true,
    color,
  });
}

export function metric(ctx, slide, label, value, x, y, color = C.blue, note = "") {
  ctx.addShape(slide, { x, y, w: 170, h: 110, fill: C.panel, line: ctx.line(C.line, 1) });
  ctx.addText(slide, { text: value, x: x + 18, y: y + 18, w: 134, h: 42, fontSize: 34, bold: true, color });
  ctx.addText(slide, { text: label, x: x + 18, y: y + 63, w: 134, h: 20, fontSize: 12, bold: true, color: C.ink });
  if (note) {
    ctx.addText(slide, { text: note, x: x + 18, y: y + 84, w: 134, h: 18, fontSize: 9, color: C.muted });
  }
}

export function row(ctx, slide, y, cols, opts = {}) {
  const xs = opts.xs || [70, 320, 565, 810, 1020];
  const ws = opts.ws || [230, 220, 220, 190, 170];
  const h = opts.h || 46;
  const fill = opts.fill || C.panel;
  ctx.addShape(slide, { x: xs[0] - 10, y, w: 1140, h, fill, line: ctx.line(opts.line || C.line, 1) });
  cols.forEach((cell, idx) => {
    ctx.addText(slide, {
      text: cell,
      x: xs[idx],
      y: y + 12,
      w: ws[idx],
      h: h - 18,
      fontSize: opts.fontSize || 12,
      bold: opts.bold || false,
      color: opts.color || C.ink,
    });
  });
}

export function bar(ctx, slide, label, value, max, x, y, w, color, note = "") {
  const barW = Math.max(8, Math.round((value / max) * w));
  ctx.addText(slide, { text: label, x, y: y - 3, w: 190, h: 22, fontSize: 12, bold: true, color: C.ink });
  ctx.addShape(slide, { x: x + 200, y, w, h: 20, fill: "#E9EDF4", line: ctx.line("#E9EDF4", 0) });
  ctx.addShape(slide, { x: x + 200, y, w: barW, h: 20, fill: color, line: ctx.line(color, 0) });
  ctx.addText(slide, { text: value.toFixed(3), x: x + 210 + w, y: y - 1, w: 70, h: 20, fontSize: 12, bold: true, color });
  if (note) {
    ctx.addText(slide, { text: note, x: x + 200, y: y + 24, w, h: 18, fontSize: 9, color: C.muted });
  }
}

export function stepBox(ctx, slide, label, x, y, w, h, fill, color = C.ink) {
  ctx.addShape(slide, { x, y, w, h, fill, line: ctx.line(C.line, 1) });
  ctx.addText(slide, {
    text: label,
    x: x + 14,
    y: y + 14,
    w: w - 28,
    h: h - 28,
    fontSize: 14,
    bold: true,
    color,
    valign: "mid",
  });
}

export function connector(ctx, slide, x, y, w, color = C.line) {
  ctx.addShape(slide, { x, y, w, h: 3, fill: color, line: ctx.line(color, 0) });
  ctx.addShape(slide, { x: x + w - 10, y: y - 5, w: 10, h: 13, fill: color, line: ctx.line(color, 0) });
}
