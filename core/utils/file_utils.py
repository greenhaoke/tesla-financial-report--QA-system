# core/utils/file_utils.py
import re
from pathlib import Path
from typing import Optional, List, Dict


def parse_filename(filename: str) -> Optional[Dict]:
    """
    从文件名解析财报元数据。

    命名规范：tesla_{report_type}_{year}[_q{quarter}].pdf
    示例：
      tesla_10k_2022.pdf         → 10-K 年报，2022 年
      tesla_10q_2023_q2.pdf      → 10-Q 季报，2023 年 Q2
      tesla_10q_2024_q3.pdf      → 10-Q 季报，2024 年 Q3

    也兼容 SEC EDGAR 常见命名（如 tsla-20231231.htm 转换而来的 PDF）

    Returns:
        dict with keys: source_file, report_type, year, quarter, fiscal_period
        or None if filename doesn't match any known pattern.
    """
    name = Path(filename).stem.lower()

    # ── 标准命名：tesla_10k_2022 / tesla_10q_2023_q2 ──────────────────
    pattern = r"tesla[_-](10[kq])[_-](\d{4})(?:[_-]q(\d))?(?:[_-].*)?"
    m = re.search(pattern, name)
    if m:
        report_type = "10-K" if "10k" in m.group(1) else "10-Q"
        year = int(m.group(2))
        quarter = int(m.group(3)) if m.group(3) else None
        fiscal_period = f"FY{year}" if report_type == "10-K" else f"{year}Q{quarter}"
        return {
            "source_file": filename,
            "report_type": report_type,
            "year": year,
            "quarter": quarter,
            "fiscal_period": fiscal_period,
        }

    # ── 兼容 tsla-YYYYMMDD 格式（SEC EDGAR 导出） ──────────────────────
    pattern2 = r"tsla[_-](\d{4})(\d{2})(\d{2})"
    m2 = re.search(pattern2, name)
    if m2:
        year = int(m2.group(1))
        month = int(m2.group(2))
        # 10-K 通常在 12 月结尾；10-Q 在 3/6/9 月
        if month == 12:
            report_type, quarter = "10-K", None
            fiscal_period = f"FY{year}"
        else:
            report_type = "10-Q"
            quarter = {3: 1, 6: 2, 9: 3}.get(month, 1)
            fiscal_period = f"{year}Q{quarter}"
        return {
            "source_file": filename,
            "report_type": report_type,
            "year": year,
            "quarter": quarter,
            "fiscal_period": fiscal_period,
        }

    # ── 兼容 tesla-{year}-{type} 变体（如 tesla-2022-annual-report） ───
    pattern3 = r"tesla[_-](\d{4})[_-](annual|10k|10-k)"
    m3 = re.search(pattern3, name)
    if m3:
        year = int(m3.group(1))
        return {
            "source_file": filename,
            "report_type": "10-K",
            "year": year,
            "quarter": None,
            "fiscal_period": f"FY{year}",
        }

    return None


def scan_pdf_files(data_dir: str) -> List[Dict]:
    """
    递归扫描 data_dir 下的所有 PDF 文件，返回带解析元数据的列表。

    跳过无法识别文件名的 PDF 并打印警告。

    Returns:
        list of dicts (parse_filename result + extra key 'file_path')
    """
    results = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"[警告] 数据目录不存在: {data_path}")
        return results

    for pdf_file in sorted(data_path.rglob("*.pdf")):
        meta = parse_filename(pdf_file.name)
        if meta is None:
            print(f"[警告] 无法解析文件名，跳过: {pdf_file.name}")
            print(f"       请将文件重命名为: tesla_10k_YYYY.pdf 或 tesla_10q_YYYY_qN.pdf")
            continue
        meta["file_path"] = str(pdf_file)
        results.append(meta)

    return results
