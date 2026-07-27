window.MathJax = {
  loader: {
    // \boldsymbol는 MathJax 기본 TeX 입력에 없어서 이 패키지를 명시적으로 안 불러오면
    // 인식 못 하고 빨간 원문 그대로 노출된다 (실제 여러 논문에서 발견됨).
    load: ["[tex]/boldsymbol"]
  },
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: { "[+]": ["boldsymbol"] }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
