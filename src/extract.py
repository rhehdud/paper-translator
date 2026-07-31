"""선정된 논문의 PDF를 내려받아 Marker로 구조화된 마크다운을 추출한다.

사용법:
    python src/extract.py --selected selected_cs_CL.json --work-dir work/cs_CL
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml
from pypdf import PdfReader


def _marker_single_binary() -> str:
    """pip/conda 환경의 bin 디렉터리에 설치된 marker_single을 찾는다 (PATH에 없어도 동작하도록)."""
    candidate = Path(sys.executable).parent / "marker_single"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("marker_single")
    if found:
        return found
    raise FileNotFoundError("marker_single 실행 파일을 찾을 수 없습니다 (marker-pdf가 설치됐는지 확인하세요)")


def download_pdf(pdf_url: str, dest: Path) -> None:
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "paper-translator-bot/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def count_pages(pdf_path: Path) -> int:
    return len(PdfReader(str(pdf_path)).pages)


def normalize_citation_brackets(markdown_text: str) -> str:
    """Marker가 인용 표기 "(Liu et al., 2025)"나 "[Kwa et al., 2026]"를 마크다운
    링크 텍스트 안에서 이스케이프할 때 \\(, \\), \\[, \\] (백슬래시 1~2개) 형태로
    내보내는 경우가 있는데, 이건 우연히 MathJax의 인라인/디스플레이 수식 구분자
    (\\( \\), \\[ \\])와 똑같이 생겼다. 그래서 pymdownx.arithmatex가 평범한 인용
    텍스트를 수식으로 오인해 감싸버리고, MathJax가 그 내용을 수식으로 파싱하려다
    실패해 "You can't use '#' in math mode" 같은 에러가 그대로 페이지에 노출된다
    (실제 발견됨). 이 이스케이프는 애초에 불필요하므로 -- 괄호는 마크다운에서 이스케이프가
    필요 없고, 대괄호는 HTML 엔티티로 대체 가능 -- 걷어낸다."""
    markdown_text = re.sub(r"\\{1,2}\(", "(", markdown_text)
    markdown_text = re.sub(r"\\{1,2}\)", ")", markdown_text)
    markdown_text = re.sub(r"\\{1,2}\[", "&#91;", markdown_text)
    markdown_text = re.sub(r"\\{1,2}\]", "&#93;", markdown_text)
    return markdown_text


# 정상적으로 짝이 맞는 $$...$$/$...$ 구간을 찾는 패턴 (translate.py의 MATH_SPAN_RE와 동일 기준).
MATH_SPAN_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)


def escape_orphan_dollars(markdown_text: str) -> str:
    """참고문헌 제목 등에 "$200k"처럼 화폐 단위로 쓰인 단독 $ 기호가 섞여 있으면,
    수식이 아닌데도 수식 시작 기호로 오인되어 그 뒤로 문서 전체의 $...$/$$...$$ 짝
    인식이 밀려버리고 렌더링이 통째로 깨진다. 정상적으로 짝이 맞는 구간을 먼저 찾고,
    거기 포함되지 않은 나머지 $는 모두 \\$로 이스케이프해 리터럴 문자로 처리한다."""
    out: list[str] = []
    last_end = 0
    for m in MATH_SPAN_RE.finditer(markdown_text):
        out.append(markdown_text[last_end : m.start()].replace("$", "\\$"))
        out.append(m.group(0))
        last_end = m.end()
    out.append(markdown_text[last_end:].replace("$", "\\$"))
    return "".join(out)


HEADING_LINE_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$")
# 번호 앞에 <span id="..."></span> 앵커나 **/* 강조 마커가 붙는 경우가 있어 건너뛰고 매칭한다.
# 번호 뒤에도 "**4.2.** 제목"처럼 닫는 강조 마커가 공백 앞에 올 수 있어 마찬가지로 건너뛴다
# (실제로 이 경우만 못 잡아서 4.2가 형제 섹션과 다른 레벨로 나온 사례 발견됨).
NUMBERED_HEADING_RE = re.compile(
    r"^(?:<span[^>]*>\s*</span>\s*)*(?:\*{1,2}|_{1,2})?(\d+(?:\.\d+)*)\.?(?:\*{1,2}|_{1,2})?\s"
)


def normalize_heading_levels(markdown_text: str) -> str:
    """Marker가 PDF 안 글꼴 크기만 보고 헤딩 레벨(#의 개수)을 추측하다 보니, 같은
    깊이의 절이 서로 다른 레벨로 뽑히는 경우가 흔하다 (예: "2.2.2"는 H2인데 바로
    다음 "2.2.3"은 H4로 뽑혀 같은 깊이인데 글씨 크기가 달라짐). 논문 절 제목에는
    이미 "2.2.3"처럼 번호가 붙어 있으므로, 그 번호의 점 개수(계층 깊이)로 진짜
    레벨을 계산해 Marker가 추측한 레벨을 덮어쓴다. 번호가 없는 헤딩(Abstract,
    References 등)은 판단할 근거가 없으므로 그대로 둔다."""
    lines = markdown_text.split("\n")
    fixed: list[str] = []
    for line in lines:
        heading_match = HEADING_LINE_RE.match(line)
        if not heading_match:
            fixed.append(line)
            continue
        text = heading_match.group(2)
        num_match = NUMBERED_HEADING_RE.match(text)
        if not num_match:
            fixed.append(line)
            continue
        depth = num_match.group(1).count(".") + 1
        level = min(depth + 1, 6)  # H1은 publish.py가 붙이는 페이지 제목 전용
        fixed.append("#" * level + " " + text)
    return "\n".join(fixed)


PSEUDOCODE_STEP_RE = re.compile(r"^\s*\d+:\s")


def _is_pseudocode_paragraph(paragraph: str) -> bool:
    stripped = paragraph.strip()
    if stripped.startswith("```") and stripped.endswith("```") and len(stripped) >= 6:
        inner = stripped[3:-3]
        return any(PSEUDOCODE_STEP_RE.match(line) for line in inner.split("\n"))
    first_line = stripped.split("\n", 1)[0]
    return bool(PSEUDOCODE_STEP_RE.match(first_line))


def _strip_code_fence(paragraph: str) -> str:
    stripped = paragraph.strip()
    if stripped.startswith("```") and stripped.endswith("```") and len(stripped) >= 6:
        return stripped[3:-3].strip("\n")
    return stripped


def normalize_pseudocode_blocks(markdown_text: str) -> str:
    """Marker가 "1: ...", "2: ..."처럼 번호 붙은 의사코드 단계를 뽑을 때, 첫 줄만
    코드 블록(```)으로 감싸고 나머지 단계는 각각 독립된 문단으로 흩어놓는 경우가
    있다. 이러면 코드 서식이 깨질 뿐 아니라, 문단 안 LaTeX 명령어가 $ 없이 그대로
    노출되어 렌더링이 지저분해진다. "N:"으로 시작하는 문단(또는 그런 줄을 담은
    코드 블록)이 연속으로 나오면 하나의 코드 블록으로 합친다."""
    paragraphs = markdown_text.split("\n\n")
    out: list[str] = []
    i = 0
    n = len(paragraphs)
    while i < n:
        if _is_pseudocode_paragraph(paragraphs[i]):
            block: list[str] = []
            while i < n and _is_pseudocode_paragraph(paragraphs[i]):
                block.append(_strip_code_fence(paragraphs[i]))
                i += 1
            out.append("```\n" + "\n".join(block) + "\n```")
            continue
        out.append(paragraphs[i])
        i += 1
    return "\n\n".join(out)


TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def normalize_table_spacing(markdown_text: str) -> str:
    """Marker가 표 캡션·본문 바로 다음 줄에 빈 줄 없이 표를 이어붙이면, 마크다운 표
    파서가 표의 시작을 인식하지 못해 파이프(|) 문자가 렌더링되지 않고 그대로 텍스트로
    노출되는 버그가 생긴다. 표로 보이는 줄(파이프로 시작·끝)의 앞뒤에 빈 줄이 없으면
    강제로 넣어 항상 별도 블록으로 분리한다."""
    lines = markdown_text.split("\n")
    fixed: list[str] = []
    for i, line in enumerate(lines):
        is_table_row = bool(TABLE_ROW_RE.match(line))
        prev_is_table_row = bool(fixed) and bool(TABLE_ROW_RE.match(fixed[-1]))
        if is_table_row and fixed and fixed[-1].strip() != "" and not prev_is_table_row:
            fixed.append("")
        fixed.append(line)
        if is_table_row:
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            next_is_table_row = next_line is not None and bool(TABLE_ROW_RE.match(next_line))
            if next_line is not None and next_line.strip() != "" and not next_is_table_row:
                fixed.append("")
    return "\n".join(fixed)


def normalize_anchor_spacing(markdown_text: str) -> str:
    """Marker가 <span id="..."></span> 앵커 태그 바로 다음 줄에 빈 줄 없이 내용을 붙여두면,
    마크다운이 그 뒤에 오는 $$ 수식을 같은 문단으로 묶어버려 표시(display) 수식이 인라인으로
    잘못 처리되면서 '$' 기호가 그대로 남는 렌더링 버그가 생긴다. 앵커 단독 줄 뒤에 빈 줄을
    강제로 넣어 항상 별도 문단으로 분리한다."""
    lines = markdown_text.split("\n")
    fixed: list[str] = []
    for i, line in enumerate(lines):
        fixed.append(line)
        if re.fullmatch(r'\s*<span id="[^"]*"></span>\s*', line):
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            if next_line is not None and next_line.strip() != "":
                fixed.append("")
    return "\n".join(fixed)


# $$...$$는 여러 줄에 걸쳐 있을 수 있어서 줄 단위가 아니라 블록(DOTALL) 단위로 찾는다.
EQUATION_BLOCK_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
# 수식 블록 바로 뒤에 "(2)"처럼 같은 줄에 붙은 수식 번호 (괄호 안 10자 이내로 제한해 오탐 방지)
EQUATION_TRAILING_LABEL_RE = re.compile(r"^[ \t]*(\([^()\n]{1,10}\))")


def _ensure_trailing_blank(text: str) -> str:
    """문자열이 실제 내용으로 끝난다면, 끝에 문단 구분용 빈 줄(개행 2개)이 오도록
    보장한다 (문서 맨 앞처럼 내용이 없으면 그대로 둔다)."""
    if not text.strip():
        return text
    return text.rstrip("\n") + "\n\n"


def _ensure_leading_blank(text: str) -> str:
    """문자열이 실제 내용으로 시작한다면, 앞에 문단 구분용 빈 줄이 오도록 보장한다
    (문서 맨 끝처럼 내용이 없으면 그대로 둔다)."""
    if not text.strip():
        return text
    return "\n\n" + text.lstrip("\n")


def normalize_equation_spacing(markdown_text: str) -> str:
    """Marker가 디스플레이 수식($$...$$, 한 줄일 수도 여러 줄일 수도 있음) 바로 옆(같은
    줄 끝 또는 바로 다음 줄)에 빈 줄 없이 "(2)" 같은 수식 번호를 붙여두면, 마크다운이
    수식과 번호를 하나의 문단으로 묶어버려 표시 수식이 인라인으로 잘못 처리되는 렌더링
    버그가 생긴다 (앵커 태그와 같은 클래스의 문제). 같은 줄에 붙은 번호는 먼저 별도
    줄로 떼어내고, 수식 블록 앞뒤에 빈 줄이 없으면 강제로 넣어 항상 별도 블록으로
    분리한다."""
    segments = EQUATION_BLOCK_RE.split(markdown_text)
    equations = EQUATION_BLOCK_RE.findall(markdown_text)

    out: list[str] = [segments[0]]
    for i, eq in enumerate(equations):
        rest = segments[i + 1]

        label_match = EQUATION_TRAILING_LABEL_RE.match(rest)
        label = ""
        if label_match:
            label = label_match.group(1)
            rest = rest[label_match.end():]

        out[-1] = _ensure_trailing_blank(out[-1])
        out.append(eq)

        tail = f"\n\n{label}" if label else ""
        tail += _ensure_leading_blank(rest)
        out.append(tail)
    return "".join(out)


def run_marker(pdf_path: Path, output_dir: Path, llm_config: dict | None = None) -> Path:
    """llm_config가 있으면 marker의 --use_llm으로 표·수식 등을 페이지 이미지 기반으로
    재해석시킨다. NVIDIA_API_KEY 환경변수가 없으면(로컬 실험 등) 조용히 건너뛴다."""
    cmd = [
        _marker_single_binary(),
        str(pdf_path),
        "--output_dir",
        str(output_dir),
        "--output_format",
        "markdown",
    ]
    if llm_config:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if api_key:
            cmd += [
                "--use_llm",
                "--llm_service",
                "marker.services.openai.OpenAIService",
                "--openai_base_url",
                llm_config["base_url"],
                "--openai_model",
                llm_config["model"],
                "--openai_api_key",
                api_key,
            ]
    subprocess.run(cmd, check=True)
    stem = pdf_path.stem
    md_path = output_dir / stem / f"{stem}.md"
    if not md_path.exists():
        raise FileNotFoundError(f"Marker 출력을 찾을 수 없습니다: {md_path}")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", required=True, help="collect.py가 만든 논문 1편의 JSON")
    parser.add_argument("--work-dir", required=True, help="PDF·마커 출력물을 둘 작업 디렉터리")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-pages", type=int, default=None, help="지정 안 하면 config.yaml의 extraction.max_pages를 씀")
    parser.add_argument("--out-dir", required=True, help="마크다운+이미지를 저장할 디렉터리")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    if args.max_pages is None:
        args.max_pages = full_config["extraction"]["max_pages"]

    llm_config = None
    if full_config["extraction"].get("use_llm"):
        llm_config = {
            "base_url": full_config["translation"]["base_url"],
            "model": full_config["extraction"]["llm_model"],
        }

    with open(args.selected, encoding="utf-8") as f:
        paper = json.load(f)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = work_dir / f"{paper['id']}.pdf"
    download_pdf(paper["pdf_url"], pdf_path)

    num_pages = count_pages(pdf_path)
    if num_pages > args.max_pages:
        print(
            f"건너뜀: {pdf_path.name}은 {num_pages}페이지로 max_pages({args.max_pages})를 초과합니다. "
            "이 논문은 이번 주 번역 대상에서 제외합니다.",
            file=sys.stderr,
        )
        return

    md_path = run_marker(pdf_path, work_dir, llm_config)
    marker_output_dir = md_path.parent  # Marker가 마크다운과 이미지를 같이 뽑아둔 폴더

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in marker_output_dir.iterdir():
        if item.suffix == ".json":
            continue  # _meta.json은 필요 없음
        shutil.copy(item, out_dir / item.name)

    # 마크다운 파일 이름을 다운스트림에서 예측 가능하게 고정하면서, 앵커 태그 뒤 수식이
    # 인라인으로 잘못 처리되는 걸 막기 위해 빈 줄 정규화를 적용한다.
    raw_md = (out_dir / md_path.name).read_text(encoding="utf-8")
    (out_dir / md_path.name).unlink()
    normalized_md = normalize_equation_spacing(
        normalize_anchor_spacing(
            normalize_table_spacing(
                normalize_heading_levels(
                    normalize_pseudocode_blocks(escape_orphan_dollars(normalize_citation_brackets(raw_md)))
                )
            )
        )
    )
    (out_dir / "extracted.md").write_text(normalized_md, encoding="utf-8")

    print(f"추출 완료: {out_dir} (이미지 포함)", file=sys.stderr)


if __name__ == "__main__":
    main()
