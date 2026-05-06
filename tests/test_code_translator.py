"""
Tests for data/code_translator.py

Uses mock lookup dicts — no CSV files required.
Prints a translation table for visual inspection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.code_translator import MEDSCodeTranslator


def make_translator() -> MEDSCodeTranslator:
    """Build a translator with small in-memory lookup dicts."""
    lab_lookup = {50882: "Bicarbonate", 51221: "Hemoglobin", 50912: "Creatinine"}
    diag_lookup = {"9_25000": "Diabetes mellitus without complication", "10_E119": "Type 2 diabetes mellitus without complications"}
    proc_lookup = {"9_9904": "Transfusion of packed cells"}
    return MEDSCodeTranslator(
        lab_lookup=lab_lookup,
        diag_lookup=diag_lookup,
        proc_lookup=proc_lookup,
    )


def test_lab_with_unit():
    t = make_translator()
    result = t.translate("LAB//50882//mEq/L")
    assert "Bicarbonate" in result, f"Expected 'Bicarbonate' in '{result}'"
    assert "mEq/L" in result, f"Expected unit in '{result}'"
    print(f"[test_lab_with_unit] '{result}'  PASS")


def test_lab_without_unit():
    t = make_translator()
    result = t.translate("LAB//51221")
    assert "Hemoglobin" in result, f"Expected 'Hemoglobin' in '{result}'"
    print(f"[test_lab_without_unit] '{result}'  PASS")


def test_lab_unknown_id():
    t = make_translator()
    result = t.translate("LAB//99999//mg/dL")
    assert "99999" in result or "Lab test" in result, f"Unexpected result: '{result}'"
    print(f"[test_lab_unknown_id] '{result}'  PASS")


def test_diagnosis_icd9():
    t = make_translator()
    result = t.translate("DIAGNOSIS//ICD//9//25000")
    assert "Diabetes" in result, f"Expected 'Diabetes' in '{result}'"
    print(f"[test_diagnosis_icd9] '{result}'  PASS")


def test_diagnosis_icd10():
    t = make_translator()
    result = t.translate("DIAGNOSIS//ICD//10//E119")
    assert "Type 2" in result or "diabetes" in result.lower(), f"Unexpected: '{result}'"
    print(f"[test_diagnosis_icd10] '{result}'  PASS")


def test_procedure():
    t = make_translator()
    result = t.translate("PROCEDURE//ICD//9//9904")
    assert "Transfusion" in result or "packed" in result.lower(), f"Unexpected: '{result}'"
    print(f"[test_procedure] '{result}'  PASS")


def test_gender_female():
    t = make_translator()
    assert t.translate("GENDER//F") == "Female"
    print("[test_gender_female] PASS")


def test_gender_male():
    t = make_translator()
    assert t.translate("GENDER//M") == "Male"
    print("[test_gender_male] PASS")


def test_race():
    t = make_translator()
    result = t.translate("RACE//WHITE")
    assert "White" in result or "white" in result.lower(), f"Unexpected: '{result}'"
    print(f"[test_race] '{result}'  PASS")


def test_age():
    t = make_translator()
    result = t.translate("AGE")
    assert result == "Age", f"Unexpected: '{result}'"
    print(f"[test_age] '{result}'  PASS")


def test_bmi():
    t = make_translator()
    result = t.translate("BMI")
    assert "mass" in result.lower() or "bmi" in result.lower(), f"Unexpected: '{result}'"
    print(f"[test_bmi] '{result}'  PASS")


def test_unknown_code():
    t = make_translator()
    result = t.translate("SOMEUNKNOWN//XYZ")
    assert isinstance(result, str) and len(result) > 0
    print(f"[test_unknown_code] '{result}'  PASS")


def test_empty_string():
    t = make_translator()
    result = t.translate("")
    assert result == "", f"Expected empty string, got '{result}'"
    print("[test_empty_string] PASS")


def test_batch_translation():
    t = make_translator()
    codes = [
        "LAB//50882//mEq/L",
        "DIAGNOSIS//ICD//9//25000",
        "GENDER//F",
        "RACE//BLACK",
        "AGE",
        "BMI",
        "UNKNOWN//CODE",
    ]
    results = t.translate_batch(codes)
    print("\n--- Translation table ---")
    for code, desc in zip(codes, results):
        print(f"  {code:<35s} → {desc}")
    assert len(results) == len(codes)
    print("[test_batch_translation] PASS")


if __name__ == "__main__":
    test_lab_with_unit()
    test_lab_without_unit()
    test_lab_unknown_id()
    test_diagnosis_icd9()
    test_diagnosis_icd10()
    test_procedure()
    test_gender_female()
    test_gender_male()
    test_race()
    test_age()
    test_bmi()
    test_unknown_code()
    test_empty_string()
    test_batch_translation()
    print("\nAll code_translator tests passed.")
