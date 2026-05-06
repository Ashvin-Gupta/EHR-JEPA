"""
MEDS code → natural language translator.

Unlike the TextCancEHR2 translators that operate on pre-tokenised integer
streams, this translator works directly on raw MEDS code strings as they
appear in the parquet 'code' column.

Composite codes use '//' as a separator, e.g.:
  LAB//50882//mEq/L              → lookup lab item 50882
  DIAGNOSIS//ICD//9//25000       → lookup ICD-9 diagnosis 25000
  PROCEDURE//ICD//9//9904        → lookup ICD-9 procedure 9904
  GENDER//F                      → "Female"
  RACE//WHITE                    → "Race White"
  MEDICATION//START//Metformin   → "Start Metformin"
  DRG//HCFA//327//Stomach proc   → "Stomach proc" (description embedded in code)
  HOSPITAL_ADMISSION//TYPE//SRC  → "Hospital admission: TYPE via SRC"
  HOSPITAL_DISCHARGE//HOME       → "Hospital discharge: Home"
  ICU_ADMISSION//MICU            → "ICU admission: MICU"
  ICU_DISCHARGE//MICU            → "ICU discharge: MICU"
  MEDS_DEATH                     → "Patient death"
  Blood Pressure Systolic        → "Blood Pressure Systolic" (verbatim — already readable)

Lookup CSV files expected (gzip or plain):
  d_labitems.csv[.gz]       columns: itemid (int), label (str), fluid, category
  d_icd_diagnoses.csv[.gz]  columns: icd_code (str), icd_version (int/str), long_title (str)
  d_icd_procedures.csv[.gz] columns: icd_code (str), icd_version (int/str), long_title (str)

If a lookup file is not provided (None), codes of that type fall back to the
raw code string as description.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


class MEDSCodeTranslator:
    """Translates raw MEDS code strings to natural language descriptions."""

    def __init__(
        self,
        lab_lookup: Optional[Dict[int, str]] = None,
        diag_lookup: Optional[Dict[str, str]] = None,
        proc_lookup: Optional[Dict[str, str]] = None,
    ):
        self.lab_lookup: Dict[int, str] = lab_lookup or {}
        self.diag_lookup: Dict[str, str] = diag_lookup or {}
        self.proc_lookup: Dict[str, str] = proc_lookup or {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_csv_files(
        cls,
        labitems_file: Optional[str] = None,
        diagnoses_file: Optional[str] = None,
        procedures_file: Optional[str] = None,
        items_file: Optional[str] = None,
    ) -> "MEDSCodeTranslator":
        """
        Build translator from optional CSV/gzip lookup files.

        Any file argument may be None — that code category will fall back to
        the raw code string as its description.

        Expected file locations (MIMIC-IV hosp module):
          d_labitems.csv.gz       itemid, label, fluid, category
                                    → covers standard lab test item IDs
          d_items.csv.gz           itemid, label, ...
                                    → covers ICU charted items (vitals, Heart Rate,
                                      O2 sat, Blood Pressure, etc.)
                                      d_labitems takes priority on ID conflicts.
          d_icd_diagnoses.csv.gz  icd_code, icd_version, long_title
          d_icd_procedures.csv.gz icd_code, icd_version, long_title
        """
        lab_lookup: Dict[int, str] = {}
        diag_lookup: Dict[str, str] = {}
        proc_lookup: Dict[str, str] = {}

        # Load ICU chart items first so that d_labitems can override on overlap
        if items_file is not None:
            df = pd.read_csv(items_file)
            lab_lookup.update(df.set_index("itemid")["label"].to_dict())

        if labitems_file is not None:
            df = pd.read_csv(labitems_file)
            # d_labitems takes priority over d_items for standard lab IDs
            lab_lookup.update(df.set_index("itemid")["label"].to_dict())

        if diagnoses_file is not None:
            df = pd.read_csv(diagnoses_file)
            # Key: "{icd_version}_{icd_code}" — handles both ICD-9 and ICD-10
            df["_key"] = df["icd_version"].astype(str) + "_" + df["icd_code"].astype(str)
            diag_lookup = df.set_index("_key")["long_title"].to_dict()

        if procedures_file is not None:
            df = pd.read_csv(procedures_file)
            df["_key"] = df["icd_version"].astype(str) + "_" + df["icd_code"].astype(str)
            proc_lookup = df.set_index("_key")["long_title"].to_dict()

        return cls(lab_lookup=lab_lookup, diag_lookup=diag_lookup, proc_lookup=proc_lookup)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup_item_id(self, raw: str) -> str:
        """
        Try to resolve a raw string as an integer item ID against lab_lookup
        (which contains both d_labitems and d_items entries).
        Returns the label string or empty string if not found / not numeric.
        """
        try:
            return self.lab_lookup.get(int(raw), "")
        except (ValueError, TypeError):
            return ""

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def translate(self, code: str) -> str:
        """
        Translate a single MEDS code string to a natural language description.

        Falls back to the cleaned raw code string if no lookup entry is found.
        """
        if not isinstance(code, str) or not code:
            return ""

        # Codes without '//' — handle known single-token codes first,
        # then treat the rest as already human-readable verbatim strings.
        if "//" not in code:
            upper = code.upper()
            if upper == "AGE":
                return "Age"
            elif upper == "BMI":
                return "Body mass index"
            elif upper == "MEDS_DEATH":
                return "Patient death"
            elif upper == "ED_REGISTRATION":
                return "Emergency department registration"
            elif upper == "ED_OUT":
                return "Emergency department out"
            elif upper.startswith("ED_"):
                # Generic ED event
                return "Emergency department " + code[3:].lower().replace("_", " ")
            else:
                # Already human-readable (e.g. "Blood Pressure Systolic", "eGFR")
                return code

        parts = code.split("//")
        prefix = parts[0].upper()

        try:
            # ----------------------------------------------------------
            # Lab tests
            # LAB//50882 or LAB//50882//mEq/L
            # ----------------------------------------------------------
            if prefix == "LAB":
                item_id = int(parts[1])
                label = self.lab_lookup.get(item_id)
                if label:
                    unit = parts[2] if len(parts) > 2 and parts[2].strip() not in ("", "UNK") else ""
                    return f"{label} ({unit})" if unit else label
                return f"Lab test {parts[1]}"

            # ----------------------------------------------------------
            # ICD diagnoses
            # DIAGNOSIS//ICD//9//25000 or DIAGNOSIS//ICD//10//E119
            # ----------------------------------------------------------
            elif prefix == "DIAGNOSIS":
                if len(parts) >= 4 and parts[1].upper() == "ICD":
                    key = f"{parts[2]}_{parts[3]}"
                    return self.diag_lookup.get(key, f"Diagnosis {parts[3]}")
                return code

            # ----------------------------------------------------------
            # ICD procedures
            # PROCEDURE//ICD//9//9904 or PROCEDURE//ICD//10//0BH17EZ
            # ----------------------------------------------------------
            elif prefix == "PROCEDURE":
                if len(parts) >= 4 and parts[1].upper() == "ICD":
                    key = f"{parts[2]}_{parts[3]}"
                    return self.proc_lookup.get(key, f"Procedure {parts[3]}")
                return code

            # ----------------------------------------------------------
            # Demographics
            # ----------------------------------------------------------
            elif prefix == "GENDER":
                val = parts[1].upper() if len(parts) > 1 else ""
                if val == "F":
                    return "Female"
                elif val == "M":
                    return "Male"
                return f"Gender {parts[1]}"

            elif prefix == "RACE":
                val = parts[1].replace("RACE_", "").replace("_", " ").title() if len(parts) > 1 else ""
                return f"Race {val}" if val else "Race unknown"

            # ----------------------------------------------------------
            # Medications
            # MEDICATION//START//Drug Name or MEDICATION//STOP//Drug Name
            # ----------------------------------------------------------
            elif prefix == "MEDICATION":
                action = parts[1].title() if len(parts) > 1 else ""
                drug = parts[2].strip() if len(parts) > 2 else ""
                if drug and drug.upper() != "UNK":
                    return f"{action} {drug}" if action else drug
                return f"Medication {action.lower()}" if action else "Medication"

            # ----------------------------------------------------------
            # DRG codes — description is embedded in the code string
            # DRG//HCFA//327//STOMACH, ESOPHAGEAL & DUODENAL PROC W CC
            # DRG//APR//816//TOXIC EFFECTS OF NON-MEDICINAL SUBSTANCES
            # ----------------------------------------------------------
            elif prefix == "DRG":
                if len(parts) >= 4:
                    description = parts[3].strip().title()
                    return f"Drug for: {description}"
                elif len(parts) >= 3:
                    description = parts[2].strip().title()
                    return f"Drug for: {description}"
                return "Drug-related group"

            # ----------------------------------------------------------
            # Hospital / ICU admissions and discharges
            # HOSPITAL_ADMISSION//TYPE//SOURCE
            # HOSPITAL_DISCHARGE//DESTINATION
            # ICU_ADMISSION//Unit Name
            # ICU_DISCHARGE//Unit Name
            # ----------------------------------------------------------
            elif prefix == "HOSPITAL_ADMISSION":
                adm_type = parts[1].replace("_", " ").title() if len(parts) > 1 else ""
                source = parts[2].replace("_", " ").title() if len(parts) > 2 else ""
                if adm_type and source:
                    return f"Hospital admission: {adm_type} via {source}"
                elif adm_type:
                    return f"Hospital admission: {adm_type}"
                return "Hospital admission"

            elif prefix == "HOSPITAL_DISCHARGE":
                dest = parts[1].replace("_", " ").title() if len(parts) > 1 else ""
                return f"Hospital discharge: {dest}" if dest else "Hospital discharge"

            elif prefix == "ICU_ADMISSION":
                unit = parts[1].strip() if len(parts) > 1 else ""
                return f"ICU admission: {unit}" if unit else "ICU admission"

            elif prefix == "ICU_DISCHARGE":
                unit = parts[1].strip() if len(parts) > 1 else ""
                return f"ICU discharge: {unit}" if unit else "ICU discharge"

            # ----------------------------------------------------------
            # Patient death
            # ----------------------------------------------------------
            elif prefix == "MEDS_DEATH":
                return "Patient death"

            # ----------------------------------------------------------
            # INFUSION_START / INFUSION_END
            # INFUSION_START//220949  →  look up item ID in lab_lookup
            #                            e.g. "Infusion start: Dextrose 5%"
            # ----------------------------------------------------------
            elif prefix in ("INFUSION_START", "INFUSION_END"):
                action = "Infusion start" if prefix == "INFUSION_START" else "Infusion end"
                if len(parts) >= 2:
                    item_label = self._lookup_item_id(parts[1])
                    if item_label:
                        return f"{action}: {item_label}"
                return action

            # ----------------------------------------------------------
            # SUBJECT_FLUID_OUTPUT//226599//mL
            # ----------------------------------------------------------
            elif prefix == "SUBJECT_FLUID_OUTPUT":
                unit = parts[2].strip() if len(parts) > 2 else ""
                if len(parts) >= 2:
                    item_label = self._lookup_item_id(parts[1])
                    if item_label:
                        return f"Fluid output: {item_label} ({unit})" if unit else f"Fluid output: {item_label}"
                return f"Fluid output ({unit})" if unit else "Fluid output"

            # ----------------------------------------------------------
            # SUBJECT_WEIGHT_AT_INFUSION//KG
            # ----------------------------------------------------------
            elif "WEIGHT" in prefix:
                unit = parts[1].strip() if len(parts) > 1 else ""
                return f"Subject weight ({unit.lower()})" if unit else "Subject weight"

            # ----------------------------------------------------------
            # TRANSFER_TO//type//destination
            # TRANSFER_TO//ED//Emergency Department
            # TRANSFER_TO//admit//Medicine/Cardiology
            # ----------------------------------------------------------
            elif prefix == "TRANSFER_TO":
                dest = parts[-1].strip().title() if len(parts) > 1 else ""
                return f"Transfer to: {dest}" if dest else "Transfer"

            # ----------------------------------------------------------
            # HCPCS//description  (already human-readable in parts[1])
            # ----------------------------------------------------------
            elif prefix == "HCPCS":
                desc = parts[1].strip() if len(parts) > 1 else ""
                return f"HCPCS: {desc}" if desc else "HCPCS procedure"

            # ----------------------------------------------------------
            # Generic fallback: if everything after the first separator is
            # purely numeric (an unresolved item ID), return just the
            # cleaned prefix text.  Otherwise join with spaces.
            # ----------------------------------------------------------
            else:
                # Check if the second part is a pure numeric ID (unresolved)
                if len(parts) >= 2 and parts[1].strip().isdigit():
                    return prefix.replace("_", " ").title()
                return code.replace("//", " ").replace("_", " ")

        except Exception:
            return code

    def translate_batch(self, codes: list) -> list:
        """Translate a list of code strings."""
        return [self.translate(c) for c in codes]
