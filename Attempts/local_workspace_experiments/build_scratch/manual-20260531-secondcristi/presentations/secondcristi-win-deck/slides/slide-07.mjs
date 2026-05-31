import { C, bg, foot, title } from "./style.mjs";

export async function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  bg(ctx, slide);
  title(ctx, slide, "Blockers closed before final polish", "The package now fails closed on format, legality, and checkpoint availability.");
  const items = [
    ["Official IDs", "Root CSVs use valid_0001... and anomaly_... copied through from participant inputs."],
    ["Checkpoint present", "outputs/gpt_large_train12 contains config, generation config, and model.safetensors."],
    ["Required docs", "REPORT.md, README.md, LICENSE, requirements.txt are present at repo root."],
    ["Legal usage", "Official eval inputs are used only for final inference; not for training."],
    ["Fallback", "n-gram official bundle remains available if checkpoint transfer fails."],
  ];
  items.forEach(([head, body], i) => {
    const y = 170 + i * 82;
    ctx.addShape(slide, { x: 88, y, w: 36, h: 36, fill: C.green, line: ctx.line(C.green, 0) });
    ctx.addText(slide, { text: "OK", x: 94, y: y + 10, w: 24, h: 14, fontSize: 10, bold: true, color: "#FFFFFF", align: "center" });
    ctx.addText(slide, { text: head, x: 150, y: y - 2, w: 230, h: 24, fontSize: 17, bold: true, color: C.ink });
    ctx.addText(slide, { text: body, x: 150, y: y + 27, w: 880, h: 32, fontSize: 13, color: C.muted });
  });
  ctx.addShape(slide, { x: 845, y: 170, w: 290, h: 200, fill: C.panel, line: ctx.line(C.line, 1) });
  ctx.addText(slide, { text: "Final files", x: 875, y: 198, w: 230, h: 24, fontSize: 18, bold: true, color: C.ink });
  ["nextstep.csv", "completion.csv", "anomaly.csv"].forEach((f, i) => {
    ctx.addText(slide, { text: f, x: 875, y: 245 + i * 38, w: 200, h: 22, fontSize: 14, color: C.blue, bold: true });
  });
  foot(ctx, slide);
  return slide;
}
