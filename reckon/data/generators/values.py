"""Value samplers for the synthetic corpus.

Sampled from distributions chosen to look like Indian hospital billing, not from
library defaults. A corpus of `faker` names and uniform amounts teaches a model
the wrong priors: real bills have single-name patients, initials-first Telugu
conventions, room rent that tracks ward class, and a pharmacy tail that is long.

Everything here is seeded and pure with respect to the RNG passed in, so a corpus
is reproducible from its seed alone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal

from reckon.normalize import gstin_check_digit

__all__ = [
    "REGIONS",
    "STATE_GST_CODES",
    "BILINGUAL_STATES",
    "STATE_REGION",
    "WARD_RATE_BANDS",
    "sample_patient_name",
    "sample_hospital",
    "sample_insurer",
    "sample_gstin",
    "sample_policy_number",
    "sample_line_items",
    "GeneratedItem",
    "money",
]


def money(value: Decimal | int | float | str) -> Decimal:
    """Quantise to paise. Never constructed from a float."""
    return Decimal(str(value)).quantize(Decimal("0.01"))


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------

REGIONS = ("telugu", "tamil", "north", "maharashtra", "bengal", "kerala")
_REGION_WEIGHTS = (0.30, 0.13, 0.27, 0.14, 0.08, 0.08)

_GIVEN: dict[str, tuple[str, ...]] = {
    "telugu": ("Aadithya", "Venkatesh", "Srinivas", "Lakshmi", "Padma", "Ramesh",
               "Sailaja", "Kavitha", "Naveen", "Anitha", "Rajesh", "Swapna"),
    "tamil": ("Murugan", "Kavya", "Senthil", "Meenakshi", "Arun", "Divya",
              "Karthik", "Bhuvana"),
    "north": ("Ramesh", "Sunita", "Amit", "Priya", "Vikram", "Neha", "Rohit",
              "Anjali", "Manoj", "Deepa"),
    "maharashtra": ("Sachin", "Manasi", "Nilesh", "Prajakta", "Ganesh", "Smita"),
    "bengal": ("Subhash", "Ananya", "Debashis", "Rituparna", "Arindam", "Moumita"),
    "kerala": ("Anoop", "Deepthi", "Vishnu", "Lekha", "Sreekumar", "Reshma"),
}

_SURNAME: dict[str, tuple[str, ...]] = {
    "telugu": ("Reddy", "Naidu", "Rao", "Sharma", "Chowdary", "Prasad", "Varma"),
    "tamil": ("Subramanian", "Iyer", "Pillai", "Krishnan", "Raman"),
    "north": ("Kumar", "Singh", "Gupta", "Verma", "Yadav", "Agarwal", "Mishra"),
    "maharashtra": ("Patil", "Deshmukh", "Kulkarni", "Joshi", "More"),
    "bengal": ("Banerjee", "Chatterjee", "Das", "Ghosh", "Mukherjee"),
    "kerala": ("Nair", "Menon", "Pillai", "Kurup", "Varghese"),
}

#: Honorifics are gendered, and a bill that says "Ms. Rohit Kumar / Sex: F" is a
#: tell that the corpus is synthetic. Sex is sampled first and the honorific and
#: given name follow from it.
_HONORIFICS = {
    "M": ("Mr.", "Sri", "Shri", "Dr."),
    "F": ("Mrs.", "Ms.", "Smt.", "Dr."),
}
_CHILD_HONORIFICS = {"M": ("Master", "Baby of"), "F": ("Baby of", "Kumari")}

#: Given names that read as female. Everything else in the pool reads as male.
_FEMALE_GIVEN = frozenset({
    "Lakshmi", "Padma", "Sailaja", "Kavitha", "Anitha", "Swapna", "Kavya",
    "Meenakshi", "Divya", "Bhuvana", "Sunita", "Priya", "Neha", "Anjali",
    "Deepa", "Manasi", "Prajakta", "Smita", "Ananya", "Rituparna", "Moumita",
    "Deepthi", "Lekha", "Reshma",
})


def sample_patient_name(
    rng: random.Random, sex: str, age: int, region: str | None = None
) -> tuple[str, str]:
    """Return (rendered name, region), consistent with *sex* and *age*.

    Three real conventions are represented rather than one: a single mononym,
    initials-first (``M. Aadithya Ram``, common in Telugu records), and
    given-plus-surname. Honorifics are attached at a realistic rate because the
    normalizer has to strip them.
    """
    region = region or rng.choices(REGIONS, weights=_REGION_WEIGHTS, k=1)[0]
    pool = [g for g in _GIVEN[region] if (g in _FEMALE_GIVEN) == (sex == "F")]
    given = rng.choice(pool or list(_GIVEN[region]))
    surname = rng.choice(_SURNAME[region])

    style = rng.random()
    if style < 0.12:
        name = given                                   # mononym
    elif style < 0.34 and region in {"telugu", "tamil"}:
        name = f"{rng.choice('ABDGKMNPRSTV')}. {given} {surname}"   # initials-first
    elif style < 0.42:
        name = f"{given} {rng.choice('ABDGKMNPRSV')}. {surname}"
    else:
        name = f"{given} {surname}"

    if age <= 12 and rng.random() < 0.6:
        name = f"{rng.choice(_CHILD_HONORIFICS[sex])} {name}"
    elif rng.random() < 0.55:
        name = f"{rng.choice(_HONORIFICS[sex])} {name}"
    return name, region


# --------------------------------------------------------------------------
# hospitals and geography
# --------------------------------------------------------------------------

#: GST state code -> (state, cities). Codes are the real GSTN state codes; the
#: hospital names built on top of them are invented.
STATE_GST_CODES: dict[str, tuple[str, tuple[str, ...]]] = {
    "36": ("Telangana", ("Hyderabad", "Warangal", "Karimnagar", "Nizamabad")),
    "37": ("Andhra Pradesh", ("Vijayawada", "Guntur", "Visakhapatnam", "Tirupati")),
    "29": ("Karnataka", ("Bengaluru", "Mysuru", "Hubballi")),
    "33": ("Tamil Nadu", ("Chennai", "Coimbatore", "Madurai")),
    "27": ("Maharashtra", ("Mumbai", "Pune", "Nagpur")),
    "07": ("Delhi", ("New Delhi",)),
    "19": ("West Bengal", ("Kolkata", "Howrah")),
    "32": ("Kerala", ("Kochi", "Thiruvananthapuram")),
}

#: States whose script this project actually has header strings for. A Telugu
#: header on a Tamil Nadu hospital would teach the model a false association, so
#: bilingual layouts are restricted to these states.
BILINGUAL_STATES: dict[str, str] = {
    "36": "telugu", "37": "telugu",
    "07": "hindi", "27": "hindi", "19": "hindi", "29": "hindi",
}

#: Name region follows the hospital's state, so a Bengali surname does not turn
#: up in a Vijayawada nursing home more often than it should.
STATE_REGION: dict[str, str] = {
    "36": "telugu", "37": "telugu", "33": "tamil", "27": "maharashtra",
    "19": "bengal", "32": "kerala", "29": "north", "07": "north",
}

_HOSPITAL_PREFIX = (
    "Sunrise", "Sri Sai", "Lotus", "Aarogya", "Meridian", "Pranaam", "Vasavi",
    "Sanjeevani", "Trinity", "Amrita Jyothi", "Bluepeak", "Nandana", "Kamakshi",
    "Silverline", "Gokulam", "Sreenidhi", "Anvaya", "Prathama",
)
_HOSPITAL_SUFFIX = {
    "multi_speciality": ("Multi-Speciality Hospital", "Super Speciality Hospital",
                         "Institute of Medical Sciences"),
    "general": ("Hospital", "Medical Centre", "Healthcare"),
    "nursing_home": ("Nursing Home", "Maternity & Nursing Home", "Poly Clinic"),
    "government": ("District Government Hospital", "Area Hospital",
                   "Community Health Centre"),
    "diagnostic_centre": ("Diagnostics", "Diagnostic Centre", "Scan Centre",
                          "Path Labs"),
}

_STREETS = ("Ring Road", "Jubilee Hills", "Gandhi Nagar", "MG Road", "Station Road",
            "Bazaar Street", "Sector 8", "Lake View Road", "Collectorate Road")


@dataclass(frozen=True)
class Hospital:
    name: str
    address: str
    city: str
    state: str
    gstin: str
    hospital_type: str
    state_code: str


def sample_gstin(rng: random.Random, state_code: str) -> str:
    """Structurally valid GSTIN with a CORRECT check digit.

    Generating invalid GSTINs would make the checksum a free giveaway that the
    document is synthetic, and would train the model on a pattern real bills do
    not have.
    """
    pan = (
        "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
        + "".join(rng.choice("0123456789") for _ in range(4))
        + rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )
    body = f"{state_code}{pan}{rng.choice('123456789')}Z"
    return body + (gstin_check_digit(body) or "0")


def sample_hospital(
    rng: random.Random,
    hospital_type: str | None = None,
    allowed_states: tuple[str, ...] | None = None,
) -> Hospital:
    hospital_type = hospital_type or rng.choice(list(_HOSPITAL_SUFFIX))
    state_code = rng.choice(list(allowed_states or STATE_GST_CODES))
    state, cities = STATE_GST_CODES[state_code]
    city = rng.choice(cities)

    if hospital_type == "government":
        name = f"{rng.choice(_HOSPITAL_SUFFIX[hospital_type])}, {city}"
    else:
        name = f"{rng.choice(_HOSPITAL_PREFIX)} {rng.choice(_HOSPITAL_SUFFIX[hospital_type])}"

    door = f"{rng.randint(1, 12)}-{rng.randint(1, 99)}-{rng.randint(1, 999)}"
    return Hospital(
        name=name,
        address=f"{door}, {rng.choice(_STREETS)}",
        city=city,
        state=state,
        gstin=sample_gstin(rng, state_code),
        hospital_type=hospital_type,
        state_code=state_code,
    )


# --------------------------------------------------------------------------
# insurance
# --------------------------------------------------------------------------

_INSURERS: tuple[tuple[str, str, str], ...] = (
    # (insurer, policy prefix, digit count)
    ("Star Health & Allied Insurance Co. Ltd.", "P/", 12),
    ("New India Assurance Co. Ltd.", "", 14),
    ("HDFC ERGO General Insurance", "HE", 10),
    ("Oriental Insurance Co. Ltd.", "OIC/", 11),
    ("Niva Bupa Health Insurance", "NB", 11),
    ("Care Health Insurance", "CHI-", 10),
    ("United India Insurance", "", 13),
    ("ICICI Lombard General Insurance", "IL", 12),
)
_TPAS = ("Medi Assist", "Paramount Health Services", "Health India TPA",
         "Vidal Health TPA", "MDIndia Health Insurance TPA", "Raksha TPA",
         "Family Health Plan Insurance TPA", "Good Health TPA")


def sample_insurer(rng: random.Random) -> tuple[str, str]:
    insurer, _, _ = rng.choice(_INSURERS)
    return insurer, rng.choice(_TPAS)


def sample_policy_number(rng: random.Random, insurer: str) -> str:
    """Policy formats differ per insurer, which is a real source of variance."""
    for name, prefix, digits in _INSURERS:
        if name == insurer:
            return prefix + "".join(rng.choice("0123456789") for _ in range(digits))
    return "".join(rng.choice("0123456789") for _ in range(12))


# --------------------------------------------------------------------------
# clinical items and money
# --------------------------------------------------------------------------

#: Ward class -> (min, max) daily room rent. Room rent tracks ward class, so a
#: model that learns the correlation is learning something true about bills.
WARD_RATE_BANDS: dict[str, tuple[int, int]] = {
    "general": (600, 1800),
    "semi_private": (1500, 3500),
    "private": (3000, 7000),
    "deluxe": (6000, 14000),
    "suite": (12000, 30000),
    "icu": (8000, 25000),
    "iccu": (9000, 26000),
    "nicu": (7000, 22000),
    "hdu": (5000, 12000),
    "day_care": (500, 2000),
    "emergency": (1000, 4000),
}

_WARD_LABELS: dict[str, tuple[str, ...]] = {
    "general": ("General Ward", "General", "GENERAL WARD", "Gen. Ward"),
    "semi_private": ("Semi-Private", "Semi Private Room", "Twin Sharing", "Semi-Pvt"),
    "private": ("Private", "Private Room", "Single Room", "Pvt. Room"),
    "deluxe": ("Deluxe Room", "Deluxe", "Delux Room"),
    "suite": ("Suite", "Suite Room"),
    "icu": ("ICU", "Intensive Care Unit", "I.C.U."),
    "iccu": ("ICCU", "Coronary Care Unit"),
    "nicu": ("NICU", "Neonatal ICU"),
    "hdu": ("HDU", "High Dependency Unit"),
    "day_care": ("Day Care", "Daycare"),
    "emergency": ("Emergency", "Casualty"),
}

#: Generic (non-branded) drug names from public formulary vocabulary.
_DRUGS = (
    "Paracetamol 500mg", "Amoxicillin 500mg", "Ceftriaxone 1g Inj",
    "Pantoprazole 40mg", "Ondansetron 4mg Inj", "Metformin 500mg",
    "Atorvastatin 10mg", "Amlodipine 5mg", "Azithromycin 500mg",
    "Diclofenac 50mg", "Ranitidine 150mg", "Heparin 5000IU",
    "Insulin Human 40IU", "Furosemide 40mg", "Tramadol 50mg Inj",
    "Normal Saline 500ml", "Ringer Lactate 500ml", "Dextrose 5% 500ml",
    "Enoxaparin 40mg", "Piperacillin-Tazobactam 4.5g",
)
_CONSUMABLES = (
    "Surgical Gloves (pair)", "Disposable Syringe 5ml", "IV Cannula 18G",
    "IV Set", "Urine Bag", "Surgical Mask", "Cotton Roll", "Adhesive Bandage",
    "Suture Material", "Oxygen Mask", "Nebuliser Kit", "ECG Electrodes",
)
_DIAGNOSTICS = (
    "Complete Blood Count", "Liver Function Test", "Renal Function Test",
    "Serum Electrolytes", "HbA1c", "Lipid Profile", "Thyroid Profile",
    "Urine Routine", "Blood Culture", "CRP Quantitative", "D-Dimer",
    "Prothrombin Time", "Arterial Blood Gas",
)
_RADIOLOGY = (
    "X-Ray Chest PA", "X-Ray Abdomen Erect", "USG Abdomen & Pelvis",
    "CT Brain Plain", "CT Chest Contrast", "MRI Brain Plain",
    "MRI Lumbar Spine", "2D Echocardiography", "Doppler Study Lower Limb",
)
_SURGERY = (
    "Laparoscopic Cholecystectomy", "Appendicectomy", "Caesarean Section",
    "Hernia Repair", "Coronary Angiography", "Angioplasty with Stent",
    "Total Knee Replacement", "Cataract Surgery (Phaco)", "TURP",
)
_PROFESSIONAL = (
    "Consultant Visit", "Senior Consultant Visit", "Surgeon Fee",
    "Anaesthetist Fee", "Assistant Surgeon Fee", "Physiotherapy Session",
    "Dietician Consultation",
)
_EQUIPMENT = (
    "Ventilator Charges", "Oxygen Charges", "Cardiac Monitor",
    "Infusion Pump", "Nebulisation", "BiPAP Support", "Syringe Pump",
)
_ADMIN = (
    "Registration Fee", "Admission Charges", "Medical Records Charge",
    "Discharge Summary Charge", "Documentation Charges",
)
_NON_MEDICAL = (
    "Attendant Charges", "Telephone Charges", "Television Charges",
    "Toiletries Kit", "Food & Beverages (attendant)", "Laundry Charges",
    "Ambulance Charges (non-emergency)",
)
_NURSING = ("Nursing Charges", "Nursing Care - ICU", "Injection Administration")

#: category -> (item pool, (rate low, rate high), long-tailed?)
_CATALOGUE: dict[str, tuple[tuple[str, ...], tuple[int, int], bool]] = {
    "pharmacy": (_DRUGS, (12, 2400), True),
    "consumables": (_CONSUMABLES, (15, 900), True),
    "diagnostics": (_DIAGNOSTICS, (150, 2500), False),
    "radiology": (_RADIOLOGY, (350, 12000), False),
    "surgery": (_SURGERY, (12000, 220000), False),
    "professional_fees": (_PROFESSIONAL, (400, 9000), False),
    "equipment": (_EQUIPMENT, (400, 9000), False),
    "administrative": (_ADMIN, (100, 1200), False),
    "non_medical": (_NON_MEDICAL, (80, 1500), False),
    "nursing": (_NURSING, (300, 3000), False),
}

#: How often each category appears at all. Pharmacy and consumables dominate a
#: real itemised bill by row count, which is why line-item recall is the metric
#: that matters most.
_CATEGORY_WEIGHTS: dict[str, float] = {
    "pharmacy": 0.30, "consumables": 0.18, "diagnostics": 0.14,
    "radiology": 0.07, "professional_fees": 0.09, "equipment": 0.06,
    "nursing": 0.05, "administrative": 0.05, "non_medical": 0.04, "surgery": 0.02,
}

#: IRDAI List I: never payable. The generator marks them so adjudication has
#: something real to find.
NON_PAYABLE_CATEGORIES = frozenset({"non_medical", "administrative"})


@dataclass(frozen=True)
class GeneratedItem:
    description: str
    category: str
    quantity: Decimal
    unit_rate: Decimal
    amount: Decimal
    is_payable: bool
    hsn_code: str | None


def _rate(rng: random.Random, low: int, high: int, long_tail: bool) -> Decimal:
    if long_tail:
        # Lognormal-ish: most rows cheap, a few expensive. A uniform draw here
        # would make every pharmacy row about the same size, which is not what a
        # pharmacy sub-bill looks like.
        span = rng.random() ** 2.2
    else:
        span = rng.random()
    value = low + span * (high - low)
    return money(round(value, -1) if value > 500 else round(value, 0))


def sample_ward(rng: random.Random) -> tuple[str, str]:
    """Return (canonical ward, one of its real-world spellings)."""
    ward = rng.choices(
        list(WARD_RATE_BANDS),
        weights=(0.26, 0.22, 0.16, 0.07, 0.02, 0.13, 0.03, 0.03, 0.04, 0.02, 0.02),
        k=1,
    )[0]
    return ward, rng.choice(_WARD_LABELS[ward])


def sample_line_items(
    rng: random.Random,
    *,
    ward: str,
    stay_days: int,
    n_items: int,
    include_surgery: bool,
) -> list[GeneratedItem]:
    """A plausible itemised bill for one admission."""
    items: list[GeneratedItem] = []

    low, high = WARD_RATE_BANDS[ward]
    room_rate = money(round(rng.uniform(low, high), -1))
    items.append(
        GeneratedItem(
            description=f"Room Rent - {ward.replace('_', ' ').title()}",
            category="room_rent",
            quantity=Decimal(stay_days),
            unit_rate=room_rate,
            amount=money(room_rate * stay_days),
            is_payable=True,
            hsn_code="996311" if rng.random() < 0.4 else None,
        )
    )

    if include_surgery:
        pool, (low, high), tail = _CATALOGUE["surgery"]
        rate = _rate(rng, low, high, tail)
        items.append(GeneratedItem(rng.choice(pool), "surgery", Decimal(1), rate,
                                   rate, True, None))

    categories = list(_CATEGORY_WEIGHTS)
    weights = [_CATEGORY_WEIGHTS[c] for c in categories]
    for _ in range(max(0, n_items - len(items))):
        category = rng.choices(categories, weights=weights, k=1)[0]
        pool, (low, high), tail = _CATALOGUE[category]
        rate = _rate(rng, low, high, tail)
        quantity = Decimal(rng.choices([1, 1, 1, 2, 2, 3, 4, 5, 6, 10],
                                       k=1)[0]) if category in {"pharmacy", "consumables"} \
            else Decimal(rng.choices([1, 1, 1, 2, stay_days], k=1)[0])
        items.append(
            GeneratedItem(
                description=rng.choice(pool),
                category=category,
                quantity=quantity,
                unit_rate=rate,
                amount=money(rate * quantity),
                is_payable=category not in NON_PAYABLE_CATEGORIES,
                hsn_code=(
                    "3004" if category == "pharmacy" and rng.random() < 0.5 else None
                ),
            )
        )
    return items
