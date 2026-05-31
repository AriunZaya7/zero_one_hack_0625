import { C, bg, foot, title } from "./style.mjs";

export async function slide08(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  title(ctx, slide, "Two-minute demo: show trust first, score second", "The pitch should make judges comfortable that the result is legal, reproducible, and OOD-aware.");
  const steps = [
    ["0:00-0:20", "Open with the final branch", "Show root CSVs, REPORT.md, and checkpoint path."],
    ["0:20-0:45", "Explain train-12/test-3", "Point to SECONDCRISTI manifest and family-wise OOD average."],
    ["0:45-1:15", "Compare baseline vs GPT", "Show n-gram Top-1 0.662 vs GPT Top-1 0.722 on OOD3."],
    ["1:15-1:40", "Show anomaly safety", "Explain rule validator and official rule strings."],
    ["1:40-2:00", "Close on submission readiness", "Official IDs, final CSVs, fallback, no eval training leakage."],
  ];
  steps.forEach(([time, head, body], i) => {
    const x = 86 + i * 228;
    ctx.addShape(slide, { x, y: 205, w: 190, h: 255, fill: i % 2 === 0 ? C.panel : C.blueSoft, line: ctx.line(C.line, 1) });
    ctx.addText(slide, { text: time, x: x + 18, y: 228, w: 150, h: 22, fontSize: 14, bold: true, color: i % 2 === 0 ? C.blue : C.ink });
    ctx.addText(slide, { text: head, x: x + 18, y: 276, w: 150, h: 52, fontSize: 17, bold: true, color: C.ink });
    ctx.addText(slide, { text: body, x: x + 18, y: 355, w: 150, h: 82, fontSize: 12, color: C.muted });
  });
  ctx.addShape(slide, { x: 110, y: 535, w: 960, h: 58, fill: C.dark, line: ctx.line(C.dark, 0) });
  ctx.addText(slide, {
    text: "Winning claim: SECONDCRISTI is the best legal bet because it optimizes for hidden-family generalization and ships a complete, reproducible final package.",
    x: 145,
    y: 552,
    w: 900,
    h: 26,
    fontSize: 16,
    bold: true,
    color: "#FFFFFF",
  });
  foot(ctx, slide);
  return slide;
}
