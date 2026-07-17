#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue
from com.sun.star.text.ControlCharacter import PARAGRAPH_BREAK
from com.sun.star.text.TextContentAnchorType import AS_CHARACTER
from com.sun.star.style.BreakType import PAGE_BEFORE


def prop(name: str, value: object) -> PropertyValue:
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


def connect() -> tuple[object, subprocess.Popen]:
    process = subprocess.Popen(
        [
            "libreoffice", "--headless", "--norestore", "--nodefault", "--nofirststartwizard",
            "--accept=socket,host=127.0.0.1,port=2083;urp;StarOffice.ComponentContext",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
    for _ in range(40):
        try:
            context = resolver.resolve("uno:socket,host=127.0.0.1,port=2083;urp;StarOffice.ComponentContext")
            desktop = context.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", context)
            return desktop, process
        except Exception:
            time.sleep(0.25)
    process.terminate()
    raise RuntimeError("Could not connect to headless LibreOffice")


def metric(article_csv: Path, method: str, name: str = "spearman") -> dict[str, str]:
    with article_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == method and row["metric"] == name:
                return row
    raise KeyError((method, name))


def ru_number(value: str) -> str:
    return f"{float(value):.4f}".replace(".", ",")


def insert_paragraph(text: object, cursor: object, value: str, style: str | None = None) -> None:
    text.insertControlCharacter(cursor, PARAGRAPH_BREAK, False)
    try:
        cursor.ParaStyleName = style or "Standard"
    except Exception:
        pass
    text.insertString(cursor, value, False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--article-csv", type=Path, required=True)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--output-docx", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    args = parser.parse_args()

    concept = metric(args.article_csv, "ConceptFAN")
    ig = metric(args.article_csv, "integrated_gradients")
    shap = metric(args.article_csv, "gradient_shap")
    short_sentence = (
        "Сравнение на PhysioNet 2012 показало более высокую межобучающую устойчивость агрегированных post-hoc "
        f"атрибуций: средняя корреляция Спирмена составила {ru_number(ig['mean'])} для Integrated Gradients и "
        f"{ru_number(shap['mean'])} для GradientSHAP."
    )
    desktop, process = connect()
    document = None
    try:
        document = desktop.loadComponentFromURL(
            uno.systemPathToFileUrl(str(args.source.resolve())), "_blank", 0, (prop("Hidden", True),)
        )
        if document is None:
            raise RuntimeError(f"LibreOffice could not open {args.source}")
        search = document.createSearchDescriptor()
        search.SearchString = "Ключевые слова:"
        found = document.findFirst(search)
        if found is not None:
            abstract_cursor = document.Text.createTextCursorByRange(found.getStart())
            document.Text.insertString(abstract_cursor, short_sentence + " ", False)

        text = document.Text
        discussion_search = document.createSearchDescriptor()
        discussion_search.SearchString = "4. Обсуждение"
        discussion = document.findFirst(discussion_search)
        if discussion is None:
            raise RuntimeError("Could not locate the discussion section insertion point")
        cursor = text.createTextCursorByRange(discussion.getStart())
        insert_paragraph(text, cursor, "3.8. Стабильность постфактум-атрибуций", "Heading 1")
        insert_paragraph(
            text,
            cursor,
            "Для post-hoc аудита использованы все 30 ранее обученных checkpoints Plain Transformer и те же 600 "
            "эпизодов замороженной тестовой выборки PhysioNet 2012, что и для ConceptFAN. Новые модели не обучались. "
            "Integrated Gradients выбран как детерминированный градиентный baseline, а GradientSHAP — как SHAP-подобный "
            "baseline с единым стратифицированным фоновым набором из 32 обучающих эпизодов.",
        )
        insert_paragraph(
            text,
            cursor,
            "Фактический вход сохранённых моделей имеет размерность 48×119: 37 физиологических значений, 37 масок, "
            "37 интервалов с последнего измерения и 8 статических признаков. Атрибуции логита суммировались по времени "
            "с сохранением знака. Основной анализ включал только каналы физиологических значений, заранее отображённые "
            "из существующих формул в пять групп. Маски и интервалы анализировались отдельно. Полученные величины являются "
            "агрегированными post-hoc атрибуциями прокси-концептов, а не внутренними концептами Transformer.",
        )
        insert_paragraph(
            text,
            cursor,
            "Постфактум-атрибуции Plain Transformer продемонстрировали более высокую межобучающую устойчивость, чем "
            "встроенные концептные вклады ConceptFAN. Для знаковых L1-нормированных пятикомпонентных представлений средняя "
            f"корреляция Спирмена составила {ru_number(ig['mean'])} для Integrated Gradients, {ru_number(shap['mean'])} "
            f"для GradientSHAP и {ru_number(concept['mean'])} для ConceptFAN. Model-level bootstrap дал 95%-е интервалы "
            "разностей, не пересекающие ноль. Это ограничивает обобщение основного отрицательного результата: архитектурная "
            "верность сама по себе не гарантирует стабильность, но отдельные post-hoc методы при данном протоколе устойчивее.",
        )

        rows = (("ConceptFAN", concept), ("Integrated Gradients", ig), ("GradientSHAP", shap))
        insert_paragraph(text, cursor, "Таблица 8 – Межобучающая устойчивость знаковых атрибуций в общей пятигрупповой системе координат")
        insert_paragraph(text, cursor, "Метод | Mean Spearman | Median | 95%-й интервал по парам")
        for label, values in rows:
            insert_paragraph(
                text,
                cursor,
                f"{label} | {ru_number(values['mean'])} | {ru_number(values['median'])} | "
                f"[{ru_number(values['ci95_low'])}; {ru_number(values['ci95_high'])}]",
            )
        insert_paragraph(text, cursor, "Ограничения сопоставления", "Heading 2")
        insert_paragraph(
            text,
            cursor,
            "Внутренние вклады ConceptFAN и post-hoc атрибуции входов Transformer являются разными вычислительными "
            "объектами. Отображение в пять групп обеспечивает общую размерность для сравнения воспроизводимости, но не "
            "делает их семантически эквивалентными. Анализ ограничен одним реальным датасетом; proxy-концепты эвристичны; "
            "GradientSHAP зависит от background, IG — от baseline, а V-only группировка оставляет процесс наблюдения и "
            "неиспользованные каналы вне пяти физиологических групп. Поэтому результат нельзя распространять на все XAI-методы.",
        )
        insert_paragraph(
            text,
            cursor,
            "Техническая проверка. Для IG медианная абсолютная ошибка completeness по checkpoints не превышала 0,001; "
            "детерминированный повтор, constant-input control, parameter-randomization control, сохранение суммы после "
            "группировки и SHA256 всех checkpoint прошли read-only верификацию.",
        )
        insert_paragraph(text, cursor, "Рис. 8 – Распределение межобучающей корреляции Спирмена для ConceptFAN, Integrated Gradients и GradientSHAP")
        cursor.BreakType = PAGE_BEFORE
        graphic = document.createInstance("com.sun.star.text.TextGraphicObject")
        graphic.GraphicURL = uno.systemPathToFileUrl(str(args.figure.resolve()))
        graphic.AnchorType = AS_CHARACTER
        graphic.Width = 16500
        graphic.Height = 7200
        text.insertControlCharacter(cursor, PARAGRAPH_BREAK, False)
        text.insertTextContent(cursor, graphic, False)

        args.output_docx.parent.mkdir(parents=True, exist_ok=True)
        document.storeAsURL(
            uno.systemPathToFileUrl(str(args.output_docx.resolve())),
            (prop("FilterName", "Office Open XML Text"), prop("Overwrite", True)),
        )
        document.storeToURL(
            uno.systemPathToFileUrl(str(args.output_pdf.resolve())),
            (prop("FilterName", "writer_pdf_Export"), prop("Overwrite", True)),
        )
    finally:
        if document is not None:
            document.close(True)
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
