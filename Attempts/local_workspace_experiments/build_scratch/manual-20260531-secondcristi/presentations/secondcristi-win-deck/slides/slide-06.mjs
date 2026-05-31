import { C, bg, foot, title } from "./style.mjs";

function metricGroup(ctx, slide, label, ngram, gpt, y) {
  const x = 110;
  const w = 760;
  ctx.addText(slide, { text: label, x, y: y - 6, w: 120, h: 22, fontSize: 13, bold: true, color: C.ink });
  ctx.addShape(slide, { x: x + 150, y, w, h: 16, fill: "#E9EDF4", line: ctx.line("#E9EDF4", 0) });
  ctx.addShape(slide, { x: x + 150, y, w: Math.round(w * ngram), h: 16, fill: C.amber, line: ctx.line(C.amber, 0) });
  ctx.addText(slide, { text: `n-gram ${ngram.toFixed(3)}`, x: x + 930, y: y - 4, w: 110, h: 20, fontSize: 11, color: C.amber, bold: true });
  ctx.addShape(slide, { x: x + 150, y: y + 30, w, h: 16, fill: "#E9EDF4", line: ctx.line("#E9EDF4", 0) });
  ctx.addShape(slide, { x: x + 150, y: y + 30, w: Math.round(w * gpt), h: 16, fill: C.green, line: ctx.line(C.green, 0) });
  ctx.addText(slide, { text: `GPT ${gpt.toFixed(3)}`, x: x + 930, y: y + 26, w: 110, h: 20, fontSize: 11, color: C.green, bold: true });
}

export async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  title(ctx, slide, "Evidence: GPT beats the safe baseline on OOD3", "The measured lift is on the train-12/test-3 protocol, not on the final hidden eval files.");
  metricGroup(ctx, slide, "Top-1", 0.662, 0.720, 190);
  metricGroup(ctx, slide, "Top-3", 0.937, 0.981, 280);
  metricGroup(ctx, slide, "Top-5", 0.976, 0.999, 370);
  metricGroup(ctx, slide, "MRR", 0.799, 0.847, 460);
  ctx.addShape(slide, { x: 105, y: 555, w: 330, h: 50, fill: C.greenSoft, line: ctx.line(C.greenSoft, 0) });
  ctx.addText(slide, { text: "+0.058 Top-1 lift vs n-gram", x: 130, y: 570, w: 280, h: 22, fontSize: 18, bold: true, color: C.green });
  ctx.addShape(slide, { x: 485, y: 555, w: 520, h: 50, fill: C.blueSoft, line: ctx.line(C.blueSoft, 0) });
  ctx.addText(slide, { text: "Family-wise OOD3 average: Top-1 0.7216, Top-5 0.9989, MRR 0.8474", x: 510, y: 570, w: 475, h: 22, fontSize: 15, bold: true, color: C.blue });
  foot(ctx, slide);
  return slide;
}
