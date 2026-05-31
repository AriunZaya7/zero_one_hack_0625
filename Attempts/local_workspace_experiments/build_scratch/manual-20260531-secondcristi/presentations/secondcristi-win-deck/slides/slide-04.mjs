import { C, bg, connector, foot, stepBox, title } from "./style.mjs";

export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  title(ctx, slide, "OOD protocol is explicit and judge-aligned", "SECONDCRISTI tests whether the model learned process grammar, not just known family routes.");
  stepBox(ctx, slide, "15 total families", 82, 190, 210, 92, C.panel);
  connector(ctx, slide, 292, 236, 86, C.line);
  stepBox(ctx, slide, "12 train families\nmosfet, igbt, ic\n+ scfam01-09", 378, 178, 260, 116, C.greenSoft, C.green);
  connector(ctx, slide, 638, 236, 84, C.line);
  stepBox(ctx, slide, "3 held-out OOD families\nscfam10, scfam11, scfam12", 722, 178, 292, 116, C.blueSoft, C.blue);
  connector(ctx, slide, 1014, 236, 72, C.line);
  stepBox(ctx, slide, "Report family-wise average", 1086, 190, 126, 92, C.panel);
  ctx.addShape(slide, { x: 100, y: 385, w: 500, h: 120, fill: C.panel, line: ctx.line(C.line, 1) });
  ctx.addText(slide, { text: "Why this matters", x: 128, y: 414, w: 420, h: 24, fontSize: 18, bold: true, color: C.ink });
  ctx.addText(slide, {
    text: "A hidden fourth family rewards transfer. The local score must therefore penalize family memorization before the final submission is frozen.",
    x: 128,
    y: 455,
    w: 410,
    h: 52,
    fontSize: 15,
    color: C.muted,
  });
  ctx.addShape(slide, { x: 680, y: 385, w: 470, h: 120, fill: C.amberSoft, line: ctx.line(C.amberSoft, 0) });
  ctx.addText(slide, { text: "Legal boundary", x: 708, y: 414, w: 380, h: 24, fontSize: 18, bold: true, color: C.amber });
  ctx.addText(slide, {
    text: "Official eval_input_valid.csv and eval_input_anomaly.csv are used only to produce final CSVs, not as training data.",
    x: 708,
    y: 455,
    w: 390,
    h: 52,
    fontSize: 15,
    color: C.ink,
  });
  foot(ctx, slide);
  return slide;
}
