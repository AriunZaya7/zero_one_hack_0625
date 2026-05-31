import { C, bg, foot, metric, pill } from "./style.mjs";

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide, "#F6F8FB");
  ctx.addShape(slide, { x: 0, y: 0, w: 1280, h: 150, fill: C.dark, line: ctx.line(C.dark, 0) });
  pill(ctx, slide, "FINAL BRANCH", 60, 48, 120, C.greenSoft, C.green);
  pill(ctx, slide, "LEGAL OOD PROTOCOL", 190, 48, 170, C.blueSoft, C.blue);
  ctx.addText(slide, {
    text: "SECONDCRISTI",
    x: 60,
    y: 182,
    w: 720,
    h: 72,
    fontSize: 54,
    bold: true,
    color: C.ink,
    typeface: ctx.fonts.title,
  });
  ctx.addText(slide, {
    text: "Train on 12 families, test on 3 unseen families, ship official-format GPT outputs.",
    x: 64,
    y: 260,
    w: 820,
    h: 50,
    fontSize: 21,
    color: C.muted,
  });
  metric(ctx, slide, "OOD3 Top-1", "0.722", 62, 370, C.blue, "family-wise avg");
  metric(ctx, slide, "OOD3 Top-5", "0.999", 252, 370, C.green, "family-wise avg");
  metric(ctx, slide, "OOD3 MRR", "0.847", 442, 370, C.cyan, "family-wise avg");
  metric(ctx, slide, "Families", "15", 632, 370, C.amber, "12 train / 3 test");
  ctx.addShape(slide, { x: 900, y: 190, w: 260, h: 330, fill: C.panel, line: ctx.line(C.line, 1) });
  ctx.addText(slide, { text: "Submission surface", x: 925, y: 218, w: 210, h: 26, fontSize: 18, bold: true, color: C.ink });
  const items = ["nextstep.csv", "completion.csv", "anomaly.csv", "outputs/gpt_large_train12", "n-gram fallback"];
  items.forEach((item, i) => {
    ctx.addShape(slide, { x: 925, y: 270 + i * 42, w: 16, h: 16, fill: C.green, line: ctx.line(C.green, 0) });
    ctx.addText(slide, { text: item, x: 952, y: 266 + i * 42, w: 175, h: 24, fontSize: 13, color: C.ink });
  });
  foot(ctx, slide);
  return slide;
}
