import { C, bar, bg, foot, title } from "./style.mjs";

export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  title(ctx, slide, "Why this branch ranks first", "The best winning chance is the strongest legal OOD result that also ships clean official-format artifacts.");
  bar(ctx, slide, "SECONDCRISTI GPT large", 0.722, 0.75, 90, 180, 620, C.green, "15 families; train 12; test 3; final CSVs and checkpoint present");
  bar(ctx, slide, "Cristi GPT large", 0.705, 0.75, 90, 245, 620, C.blue, "Strong self-eval, but weaker OOD framing than SECONDCRISTI");
  bar(ctx, slide, "KREMSIANS proxy", 0.582, 0.75, 90, 310, 620, C.amber, "Useful single-family stress test, but less robust than OOD3 average");
  bar(ctx, slide, "N-gram fallback", 0.662, 0.75, 90, 375, 620, C.cyan, "Dependency-free safety baseline; not the primary winning path");
  ctx.addShape(slide, { x: 870, y: 178, w: 300, h: 268, fill: C.panel, line: ctx.line(C.line, 1) });
  ctx.addText(slide, { text: "Selection logic", x: 900, y: 205, w: 230, h: 26, fontSize: 18, bold: true, color: C.ink });
  const reasons = [
    "Higher OOD confidence than inherited single-family proxy",
    "No eval_input_anomaly training leakage",
    "Official IDs generated from participant files",
    "Checkpoint included; fallback remains runnable",
  ];
  reasons.forEach((reason, i) => {
    ctx.addShape(slide, { x: 900, y: 252 + i * 46, w: 14, h: 14, fill: C.green, line: ctx.line(C.green, 0) });
    ctx.addText(slide, { text: reason, x: 922, y: 247 + i * 46, w: 215, h: 34, fontSize: 12, color: C.ink });
  });
  ctx.addText(slide, {
    text: "Counterargument handled: if the hidden OOD family differs from our synthetic families, the 3-family average is still a better proxy than optimizing on known-format self-eval alone.",
    x: 90,
    y: 515,
    w: 1030,
    h: 48,
    fontSize: 16,
    color: C.muted,
  });
  foot(ctx, slide);
  return slide;
}
