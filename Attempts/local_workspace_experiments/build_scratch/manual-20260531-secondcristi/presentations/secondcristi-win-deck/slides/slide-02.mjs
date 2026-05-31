import { C, bg, foot, row, title } from "./style.mjs";

export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  title(ctx, slide, "Score exactly where the jury scores", "Each official task has a generated CSV, a legal source path, and an auditable local proof.");
  row(ctx, slide, 170, ["Jury task", "Submitted file", "What it scores", "SECONDCRISTI proof", "Risk"], {
    fill: C.dark,
    color: "#FFFFFF",
    bold: true,
  });
  row(ctx, slide, 226, ["1. Next step", "nextstep.csv", "Top-1 / Top-3 / Top-5 / MRR", "GPT large OOD3 Top-1 0.722", "Low"]);
  row(ctx, slide, 282, ["2. Completion", "completion.csv", "Normalized edit distance plus exact/token/block", "Same checkpoint, official IDs", "Medium"]);
  row(ctx, slide, 338, ["3. Anomaly", "anomaly.csv", "Validity, score, violated rule name", "Rule validator, official rule strings", "Low"]);
  row(ctx, slide, 394, ["Legality", "repo evidence", "No hidden eval training", "Official eval only used for inference", "Low"]);
  row(ctx, slide, 450, ["Reproducibility", "REPORT + README", "Run path and dependencies", "Checkpoint + n-gram fallback", "Low"]);
  ctx.addShape(slide, { x: 75, y: 545, w: 1060, h: 74, fill: C.blueSoft, line: ctx.line(C.blueSoft, 0) });
  ctx.addText(slide, {
    text: "Decision: freeze the number-improvement path here. Polish presentation, not model behavior, unless a format bug is found.",
    x: 105,
    y: 568,
    w: 1000,
    h: 34,
    fontSize: 18,
    bold: true,
    color: C.blue,
  });
  foot(ctx, slide);
  return slide;
}
