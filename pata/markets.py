"""Per-market reference data used by the rules and the AE generator.

Markets are config: adding one means adding a block here plus a generator or dataset.
"""

from __future__ import annotations

# ---------------------------------------------------------------- UAE

EMIRATES = {
    "dubai": "Dubai",
    "dxb": "Dubai",
    "sharjah": "Sharjah",
    "shj": "Sharjah",
    "abu dhabi": "Abu Dhabi",
    "auh": "Abu Dhabi",
    "ajman": "Ajman",
    "ras al khaimah": "Ras Al Khaimah",
    "rak": "Ras Al Khaimah",
    "fujairah": "Fujairah",
    "umm al quwain": "Umm Al Quwain",
    "uaq": "Umm Al Quwain",
    "al ain": "Abu Dhabi",
    "دبي": "Dubai",
    "الشارقة": "Sharjah",
    "أبوظبي": "Abu Dhabi",
    "ابوظبي": "Abu Dhabi",
    "عجمان": "Ajman",
}

# Community -> emirate. Lowercase keys. Not exhaustive; enough to resolve the common cases.
AE_AREAS: dict[str, str] = {
    # Dubai
    "dubai marina": "Dubai",
    "jumeirah lake towers": "Dubai",
    "jlt": "Dubai",
    "jumeirah beach residence": "Dubai",
    "jbr": "Dubai",
    "al barsha": "Dubai",
    "al barsha 1": "Dubai",
    "al barsha 2": "Dubai",
    "al barsha 3": "Dubai",
    "al barsha south": "Dubai",
    "barsha heights": "Dubai",
    "tecom": "Dubai",
    "deira": "Dubai",
    "bur dubai": "Dubai",
    "karama": "Dubai",
    "al karama": "Dubai",
    "satwa": "Dubai",
    "al satwa": "Dubai",
    "jumeirah": "Dubai",
    "jumeirah 1": "Dubai",
    "jumeirah 2": "Dubai",
    "jumeirah 3": "Dubai",
    "umm suqeim": "Dubai",
    "al safa": "Dubai",
    "al wasl": "Dubai",
    "downtown dubai": "Dubai",
    "business bay": "Dubai",
    "dubai silicon oasis": "Dubai",
    "silicon oasis": "Dubai",
    "international city": "Dubai",
    "mirdif": "Dubai",
    "al warqa": "Dubai",
    "al mizhar": "Dubai",
    "al muhaisnah": "Dubai",
    "muhaisnah": "Dubai",
    "sonapur": "Dubai",
    "al qusais": "Dubai",
    "al quoz": "Dubai",
    "al nahda": "Dubai",
    "al twar": "Dubai",
    "al rashidiya": "Dubai",
    "al garhoud": "Dubai",
    "garhoud": "Dubai",
    "oud metha": "Dubai",
    "al jaddaf": "Dubai",
    "jumeirah village circle": "Dubai",
    "jvc": "Dubai",
    "jumeirah village triangle": "Dubai",
    "jvt": "Dubai",
    "dubai sports city": "Dubai",
    "motor city": "Dubai",
    "arabian ranches": "Dubai",
    "the springs": "Dubai",
    "the meadows": "Dubai",
    "the greens": "Dubai",
    "discovery gardens": "Dubai",
    "jebel ali": "Dubai",
    "dubai investments park": "Dubai",
    "dip": "Dubai",
    "al nahda 1": "Dubai",
    "al nahda 2": "Dubai",
    "palm jumeirah": "Dubai",
    "al rigga": "Dubai",
    "al muraqqabat": "Dubai",
    "naif": "Dubai",
    "al ras": "Dubai",
    "dubai south": "Dubai",
    "town square": "Dubai",
    "damac hills": "Dubai",
    "al furjan": "Dubai",
    # Sharjah
    "al majaz": "Sharjah",
    "al taawun": "Sharjah",
    "al khan": "Sharjah",
    "al qasimia": "Sharjah",
    "rolla": "Sharjah",
    "abu shagara": "Sharjah",
    "muwaileh": "Sharjah",
    "al nabba": "Sharjah",
    "industrial area": "Sharjah",
    "al yarmook": "Sharjah",
    "al jazzat": "Sharjah",
    "university city": "Sharjah",
    # Abu Dhabi
    "khalifa city": "Abu Dhabi",
    "khalifa city a": "Abu Dhabi",
    "khalifa city b": "Abu Dhabi",
    "mohammed bin zayed city": "Abu Dhabi",
    "mbz city": "Abu Dhabi",
    "musaffah": "Abu Dhabi",
    "mussafah": "Abu Dhabi",
    "al reem island": "Abu Dhabi",
    "reem island": "Abu Dhabi",
    "al khalidiyah": "Abu Dhabi",
    "khalidiya": "Abu Dhabi",
    "tourist club area": "Abu Dhabi",
    "al mushrif": "Abu Dhabi",
    "al bateen": "Abu Dhabi",
    "yas island": "Abu Dhabi",
    "saadiyat island": "Abu Dhabi",
    "al raha": "Abu Dhabi",
    "baniyas": "Abu Dhabi",
    "shakhbout city": "Abu Dhabi",
    "al shamkha": "Abu Dhabi",
    "hamdan street": "Abu Dhabi",
    "electra street": "Abu Dhabi",
    # Ajman
    "al nuaimiya": "Ajman",
    "al rashidiya ajman": "Ajman",
    "al jurf": "Ajman",
    "al rawda": "Ajman",
    "ajman industrial": "Ajman",
    "al nakhil": "Ajman",
    # Arabic spellings of common ones
    "البرشاء": "Dubai",
    "ديرة": "Dubai",
    "الكرامة": "Dubai",
    "النهدة": "Dubai",
    "القصيص": "Dubai",
    "المجاز": "Sharjah",
    "التعاون": "Sharjah",
    "مصفح": "Abu Dhabi",
}

# Areas that exist in more than one emirate, or that split into sub-areas a driver must know.
# Value: regex (applied to lowercased text + city hint) that resolves the ambiguity.
AE_AMBIGUOUS: dict[str, str] = {
    "al nahda": r"al nahda\s*[12]\b|dubai|sharjah|دبي|الشارقة",
    "النهدة": r"دبي|الشارقة|dubai|sharjah",
    "khalifa city": r"khalifa city\s*[ab]\b",
    "al rashidiya": r"dubai|ajman",
    "al quoz": r"al quoz\s*(?:industrial\s*(?:area\s*)?)?[1-4]\b|al quoz industrial",
    "al qusais": r"al qusais\s*(?:industrial\s*(?:area\s*)?)?[1-5]\b|al qusais industrial",
    "industrial area": r"industrial area\s*\d+|musaffah|mussafah|ajman industrial",
}

# ---------------------------------------------------------------- India

IN_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
    "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh",
    "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "orissa", "punjab",
    "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand",
    "west bengal", "delhi", "jammu and kashmir", "ladakh", "chandigarh", "puducherry",
]

# First two digits of a PIN -> state. Approximate: some prefixes straddle states.
IN_PIN_PREFIX: dict[str, str] = {}
for _p, _s in [
    ((11,), "Delhi"), ((12, 13), "Haryana"), ((14, 15), "Punjab"), ((16,), "Chandigarh"),
    ((17,), "Himachal Pradesh"), ((18, 19), "Jammu and Kashmir"),
    ((20, 21, 22, 23, 24, 25, 26, 27, 28), "Uttar Pradesh"),
    ((30, 31, 32, 33, 34), "Rajasthan"), ((36, 37, 38, 39), "Gujarat"),
    ((40, 41, 42, 43, 44), "Maharashtra"), ((45, 46, 47, 48), "Madhya Pradesh"),
    ((49,), "Chhattisgarh"), ((50,), "Telangana"), ((51, 52, 53), "Andhra Pradesh"),
    ((56, 57, 58, 59), "Karnataka"), ((60, 61, 62, 63, 64), "Tamil Nadu"),
    ((67, 68, 69), "Kerala"), ((70, 71, 72, 73, 74), "West Bengal"),
    ((75, 76, 77), "Odisha"), ((78,), "Assam"), ((79,), "North East"),
    ((80, 84, 85), "Bihar"), ((81, 82, 83), "Jharkhand"),
]:
    for _n in _p:
        IN_PIN_PREFIX[str(_n)] = _s

IN_CITIES = [
    "mumbai", "bombay", "delhi", "new delhi", "bengaluru", "bangalore", "hyderabad", "chennai",
    "madras", "kolkata", "calcutta", "pune", "ahmedabad", "surat", "jaipur", "lucknow", "kanpur",
    "nagpur", "indore", "bhopal", "patna", "vadodara", "ghaziabad", "ludhiana", "agra", "nashik",
    "noida", "gurgaon", "gurugram", "faridabad", "thane", "navi mumbai", "coimbatore", "kochi",
    "cochin", "visakhapatnam", "vijayawada", "mysore", "mysuru", "chandigarh", "guwahati",
    "bhubaneswar", "dehradun", "ranchi", "raipur", "amritsar", "varanasi", "madurai", "mangalore",
    "trivandrum", "thiruvananthapuram", "goa", "panaji", "jodhpur", "udaipur", "rajkot", "meerut",
]


def pin_to_state(pin: str) -> str:
    return IN_PIN_PREFIX.get(pin[:2], "")
