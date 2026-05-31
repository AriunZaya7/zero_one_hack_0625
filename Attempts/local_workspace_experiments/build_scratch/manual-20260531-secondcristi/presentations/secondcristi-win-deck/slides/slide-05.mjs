import { C, bg, connector, foot, stepBox, title } from "./style.mjs";

export async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  title(ctx, slide, "System design stays simple enough to trust", "The model improves sequence predictions; the validator owns hard rule compliance.");
  stepBox(ctx, slide, "Training CSVs\n15-family manifest", 92, 205, 170, 92, C.panel);
  connector(ctx, slide, 262, 250, 58, C.line);
  stepBox(ctx, slide, "Step tokenizer\none process step = one token", 320, 196, 202, 110, C.blueSoft, C.blue);
  connector(ctx, slide, 522, 250, 58, C.line);
  stepBox(ctx, slide, "GPT large\n21.47M parameters", 580, 205, 178, 92, C.greenSoft, C.green);
  connector(ctx, slide, 758, 250, 58, C.line);
  stepBox(ctx, slide, "Ranked next steps\ncompletion beam", 816, 196, 190, 110, C.panel);
  connector(ctx, slide, 1006, 250, 58, C.line);
  stepBox(ctx, slide, "Official CSVs\nroot + final_submission", 1064, 205, 145, 92, C.dark, "#FFFFFF");
  ctx.addShape(slide, { x: 145, y: 405, w: 440, h: 92, fill: C.panel, line: ctx.line(C.line, 1) });
  ctx.addText(slide, { text: "Fallback path", x: 172, y: 426, w: 180, h: 22, fontSize: 17, bold: true, color: C.ink });
  ctx.addText(slide, {
    text: "Trigram n-gram runs without torch and can regenerate official-shaped CSVs from a clean checkout.",
    x: 172,
    y: 458,
    w: 350,
    h: 34,
    fontSize: 13,
    color: C.muted,
  });
  ctx.addShape(slide, { x: 690, y: 405, w: 440, h: 92, fill: C.panel, line: ctx.line(C.line, 1) });
  ctx.addText(slide, { text: "Safety path", x: 717, y: 426, w: 180, h: 22, fontSize: 17, bold: true, color: C.ink });
  ctx.addText(slide, {
    text: "GPT runner refuses random-init submissions unless explicitly overridden, preventing silent bad CSVs.",
    x: 717,
    y: 458,
    w: 350,
    h: 34,
    fontSize: 13,
    color: C.muted,
  });
  foot(ctx, slide);
  return slide;
}
