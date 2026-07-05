from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence


BLACK = 1
WHITE = -1

SCORING_INDEX = {
    "area": 0,
    "area_ancient_chinese": 1,
    "territory": 2,
    "territory_with_seki_score": 3,
}
KO_INDEX = {
    "simple_ko": 0,
    "positional_superko": 1,
}
SUICIDE_INDEX = {
    "allowed": 0,
    "forbidden": 1,
}

SCORING_ALIASES = {
    **{key: key for key in SCORING_INDEX},
    "ancient_chinese": "area_ancient_chinese",
    "territory_seki": "territory_with_seki_score",
    "territory_with_seki": "territory_with_seki_score",
}
KO_ALIASES = {
    **{key: key for key in KO_INDEX},
    "simple": "simple_ko",
    "psk": "positional_superko",
    "positional": "positional_superko",
}
KATAGO_KO_ALIASES = {
    **KO_ALIASES,
    "situational": "situational_superko",
    "situational_superko": "situational_superko",
    "ssk": "situational_superko",
}
SUICIDE_ALIASES = {
    **{key: key for key in SUICIDE_INDEX},
    "yes": "allowed",
    "true": "allowed",
    "no": "forbidden",
    "false": "forbidden",
}


def _normalize(raw: str, aliases: Mapping[str, str], label: str) -> str:
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in aliases:
        available = ", ".join(sorted(aliases))
        raise ValueError(f"unknown {label} {raw!r}; available: {available}")
    return aliases[key]


def normalize_scoring(raw: str) -> str:
    return _normalize(raw, SCORING_ALIASES, "SAKIGo scoring rule")


def normalize_ko(raw: str) -> str:
    return _normalize(raw, KO_ALIASES, "SAKIGo ko rule")


def normalize_katago_ko(raw: str) -> str:
    return _normalize(raw, KATAGO_KO_ALIASES, "KataGo ko rule")


def normalize_suicide(raw: str) -> str:
    return _normalize(raw, SUICIDE_ALIASES, "SAKIGo suicide rule")


def parse_katago_rules(raw: str | Mapping[str, Any]) -> str | dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    text = raw.strip()
    if not text:
        raise ValueError("KataGo rules must be non-empty")
    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("--katago-rules JSON must be an object")
        return parsed
    return text


def normalize_katago_rules_payload(raw: str | Mapping[str, Any]) -> str | dict[str, Any]:
    parsed = parse_katago_rules(raw)
    if not isinstance(parsed, Mapping):
        return parsed
    normalized = dict(parsed)
    aliases = (
        ("koRule", "ko"),
        ("ko_rule", "ko"),
        ("scoringRule", "scoring"),
        ("scoring_rule", "scoring"),
        ("taxRule", "tax"),
        ("tax_rule", "tax"),
        ("multiStoneSuicideLegal", "suicide"),
        ("multi_stone_suicide_legal", "suicide"),
    )
    for old_key, new_key in aliases:
        if old_key not in normalized:
            continue
        if new_key in normalized and normalized[new_key] != normalized[old_key]:
            raise ValueError(f"KataGo rules specify conflicting {old_key!r} and {new_key!r}")
        normalized[new_key] = normalized.pop(old_key)
    if "whiteHandicapBonus" in normalized:
        normalized["whiteHandicapBonus"] = str(normalized["whiteHandicapBonus"])
    return normalized


def _string_field(raw: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str):
            return value
    return None


def _bool_field(raw: Mapping[str, Any], *keys: str) -> bool | None:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "1"):
                return True
            if lowered in ("false", "no", "0"):
                return False
    return None


def infer_katago_ko(raw: str | Mapping[str, Any]) -> str | None:
    parsed = parse_katago_rules(raw)
    if not isinstance(parsed, Mapping):
        return None
    ko = _string_field(parsed, "koRule", "ko_rule", "ko")
    return None if ko is None else normalize_katago_ko(ko)


def infer_katago_suicide(raw: str | Mapping[str, Any]) -> str | None:
    parsed = parse_katago_rules(raw)
    if not isinstance(parsed, Mapping):
        return None
    legal = _bool_field(parsed, "multiStoneSuicideLegal", "multi_stone_suicide_legal", "suicide")
    return None if legal is None else ("allowed" if legal else "forbidden")


def infer_katago_scoring(raw: str | Mapping[str, Any]) -> str | None:
    parsed = parse_katago_rules(raw)
    if not isinstance(parsed, Mapping):
        return None
    scoring = _string_field(parsed, "scoringRule", "scoring_rule", "scoring")
    if scoring is None:
        return None
    scoring_key = scoring.strip().lower().replace("-", "_").replace(" ", "_")
    tax_raw = _string_field(parsed, "taxRule", "tax_rule", "tax") or "NONE"
    tax = tax_raw.strip().lower().replace("-", "_").replace(" ", "_")
    if scoring_key == "area":
        if tax == "none":
            return "area"
        if tax == "all":
            return "area_ancient_chinese"
    if scoring_key == "territory":
        if tax == "seki":
            return "territory"
        if tax == "none":
            return "territory_with_seki_score"
    return None


@dataclass(frozen=True)
class RulesetSpec:
    """Mapping between a KataGo analysis query rule and SAKIGo rule features."""

    name: str
    katago_rules: str | dict[str, Any]
    scoring: str
    ko: str
    suicide: str
    komi: float
    katago_ko: str | None = None
    katago_suicide: str | None = None

    def __post_init__(self) -> None:
        katago_rules = normalize_katago_rules_payload(self.katago_rules)
        object.__setattr__(self, "scoring", normalize_scoring(self.scoring))
        object.__setattr__(self, "ko", normalize_ko(self.ko))
        object.__setattr__(self, "suicide", normalize_suicide(self.suicide))
        object.__setattr__(self, "katago_rules", katago_rules)
        object.__setattr__(self, "komi", float(self.komi))
        katago_ko = self.katago_ko or infer_katago_ko(katago_rules) or self.ko
        katago_suicide = self.katago_suicide or infer_katago_suicide(katago_rules) or self.suicide
        object.__setattr__(self, "katago_ko", normalize_katago_ko(katago_ko))
        object.__setattr__(self, "katago_suicide", normalize_suicide(katago_suicide))
        inferred_scoring = infer_katago_scoring(katago_rules)
        if inferred_scoring is not None and inferred_scoring != self.scoring:
            raise ValueError(
                f"KataGo scoring {inferred_scoring!r} does not exactly project to "
                f"SAKIGo scoring {self.scoring!r}"
            )
        if self.katago_ko not in KO_INDEX or self.katago_ko != self.ko:
            raise ValueError(
                f"KataGo ko {self.katago_ko!r} does not exactly project to "
                f"SAKIGo ko {self.ko!r}"
            )
        if self.katago_suicide != self.suicide:
            raise ValueError(
                f"KataGo suicide {self.katago_suicide!r} does not exactly project to "
                f"SAKIGo suicide {self.suicide!r}"
            )

    @property
    def allows_suicide(self) -> bool:
        return self.katago_suicide == "allowed"

    @property
    def uses_positional_superko(self) -> bool:
        return self.katago_ko == "positional_superko"

    @property
    def uses_situational_superko(self) -> bool:
        return self.katago_ko == "situational_superko"

    @property
    def uses_superko(self) -> bool:
        return self.uses_positional_superko or self.uses_situational_superko

    def query_fields(self) -> dict[str, Any]:
        return {
            "rules": self.katago_rules,
            "komi": self.komi,
        }

    def with_komi(self, komi: float) -> RulesetSpec:
        return replace(self, komi=float(komi))

    def rule_features(
        self,
        *,
        to_move: int,
        captures: Sequence[int],
        board_area: int,
    ) -> list[float]:
        if to_move not in (BLACK, WHITE):
            raise ValueError("to_move must be BLACK=1 or WHITE=-1")
        if len(captures) != 2:
            raise ValueError("captures must be [black_captures, white_captures]")
        area = max(float(board_area), 1.0)
        features = [0.0] * 10
        features[SCORING_INDEX[self.scoring]] = 1.0
        features[4 + KO_INDEX[self.ko]] = 1.0
        features[6 + SUICIDE_INDEX[self.suicide]] = 1.0
        signed_komi = -self.komi if to_move == BLACK else self.komi
        if to_move == BLACK:
            capture_diff = float(captures[0] - captures[1])
        else:
            capture_diff = float(captures[1] - captures[0])
        features[8] = max(-1.0, min(1.0, signed_komi / area))
        features[9] = max(-1.0, min(1.0, capture_diff / area))
        return features

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "katago_rules": self.katago_rules,
            "katago_ko": self.katago_ko,
            "katago_suicide": self.katago_suicide,
            "saki_scoring": self.scoring,
            "saki_ko": self.ko,
            "saki_suicide": self.suicide,
            "komi": self.komi,
        }

    def key(self) -> str:
        katago = json.dumps(self.katago_rules, sort_keys=True, separators=(",", ":"))
        return "|".join(
            (
                self.name,
                katago,
                str(self.katago_ko),
                str(self.katago_suicide),
                self.scoring,
                self.ko,
                self.suicide,
                f"{self.komi:g}",
            )
        )


PRESET_RULESETS: dict[str, RulesetSpec] = {
    "ancient-chinese": RulesetSpec(
        name="ancient-chinese",
        katago_rules={
            "ko": "POSITIONAL",
            "scoring": "AREA",
            "tax": "ALL",
            "suicide": True,
            "hasButton": False,
            "friendlyPassOk": False,
            "whiteHandicapBonus": "0",
        },
        scoring="area_ancient_chinese",
        ko="positional_superko",
        suicide="allowed",
        komi=7.5,
    ),
    "tromp-taylor": RulesetSpec(
        name="tromp-taylor",
        katago_rules="tromp-taylor",
        scoring="area",
        ko="positional_superko",
        suicide="allowed",
        komi=7.5,
        katago_ko="positional_superko",
        katago_suicide="allowed",
    ),
    "chinese": RulesetSpec(
        name="chinese",
        katago_rules="chinese",
        scoring="area",
        ko="simple_ko",
        suicide="forbidden",
        komi=7.5,
        katago_ko="simple_ko",
        katago_suicide="forbidden",
    ),
    "chinese-ogs": RulesetSpec(
        name="chinese-ogs",
        katago_rules="chinese-ogs",
        scoring="area",
        ko="positional_superko",
        suicide="forbidden",
        komi=7.5,
        katago_ko="positional_superko",
        katago_suicide="forbidden",
    ),
    "japanese": RulesetSpec(
        name="japanese",
        katago_rules="japanese",
        scoring="territory",
        ko="simple_ko",
        suicide="forbidden",
        komi=6.5,
        katago_ko="simple_ko",
        katago_suicide="forbidden",
    ),
    "korean": RulesetSpec(
        name="korean",
        katago_rules="korean",
        scoring="territory",
        ko="simple_ko",
        suicide="forbidden",
        komi=6.5,
        katago_ko="simple_ko",
        katago_suicide="forbidden",
    ),
}


def available_rulesets() -> tuple[str, ...]:
    return tuple(sorted(PRESET_RULESETS))


def ruleset_from_name(name: str) -> RulesetSpec:
    key = name.strip().lower().replace("_", "-")
    if key not in PRESET_RULESETS:
        available = ", ".join(available_rulesets())
        raise ValueError(f"unknown ruleset {name!r}; available: {available}")
    return PRESET_RULESETS[key]


def ruleset_from_overrides(
    *,
    ruleset: str,
    katago_rules: str | None = None,
    katago_ko: str | None = None,
    katago_suicide: str | None = None,
    saki_scoring: str | None = None,
    saki_ko: str | None = None,
    saki_suicide: str | None = None,
    komi: float | None = None,
) -> RulesetSpec:
    base = ruleset_from_name(ruleset)
    parsed_katago_rules = parse_katago_rules(katago_rules) if katago_rules else base.katago_rules
    preset_from_katago = False
    if isinstance(parsed_katago_rules, str):
        preset_key = parsed_katago_rules.strip().lower().replace("_", "-")
        matched = PRESET_RULESETS.get(preset_key)
        if matched is not None:
            base = matched
            preset_from_katago = True
        elif katago_rules:
            available = ", ".join(available_rulesets())
            raise ValueError(
                f"KataGo rules string {parsed_katago_rules!r} has no exact SAKIGo mapping; "
                f"use one of: {available}, or pass a JSON object with exact SAKIGo overrides"
            )
    mapping_override = any(
        value not in (None, "")
        for value in (
            katago_ko,
            katago_suicide,
            saki_scoring,
            saki_ko,
            saki_suicide,
            komi,
        )
    )
    has_override = katago_rules not in (None, "") or mapping_override
    name = base.name if not has_override or (preset_from_katago and not mapping_override) else f"{base.name}+custom"
    return replace(
        base,
        name=name,
        katago_rules=parsed_katago_rules,
        katago_ko=normalize_katago_ko(katago_ko) if katago_ko else infer_katago_ko(parsed_katago_rules) or base.katago_ko,
        katago_suicide=normalize_suicide(katago_suicide)
        if katago_suicide
        else infer_katago_suicide(parsed_katago_rules) or base.katago_suicide,
        scoring=normalize_scoring(saki_scoring) if saki_scoring else base.scoring,
        ko=normalize_ko(saki_ko) if saki_ko else base.ko,
        suicide=normalize_suicide(saki_suicide) if saki_suicide else base.suicide,
        komi=base.komi if komi is None else float(komi),
    )


def ruleset_from_metadata(raw: Any) -> RulesetSpec | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return ruleset_from_name(raw)
    if not isinstance(raw, Mapping):
        raise ValueError("ruleset metadata must be a string or object")
    try:
        name = str(raw.get("name", "custom"))
        katago_rules = raw["katago_rules"]
        katago_ko = raw.get("katago_ko")
        katago_suicide = raw.get("katago_suicide")
        scoring = str(raw["saki_scoring"])
        ko = str(raw["saki_ko"])
        suicide = str(raw["saki_suicide"])
        komi = float(raw["komi"])
    except KeyError as exc:
        raise ValueError(f"ruleset metadata is missing {exc}") from exc
    return RulesetSpec(
        name=name,
        katago_rules=katago_rules,
        scoring=scoring,
        ko=ko,
        suicide=suicide,
        komi=komi,
        katago_ko=str(katago_ko) if katago_ko is not None else None,
        katago_suicide=str(katago_suicide) if katago_suicide is not None else None,
    )


def ruleset_key_from_raw(raw: Any) -> str:
    ruleset = ruleset_from_metadata(raw)
    return "" if ruleset is None else ruleset.key()


def _active_one_hot_index(values: Sequence[float], label: str) -> int:
    if len(values) == 0:
        raise ValueError(f"{label} must not be empty")
    active = [index for index, value in enumerate(values) if abs(float(value) - 1.0) <= 1e-5]
    inactive_ok = all(
        abs(float(value)) <= 1e-5 or abs(float(value) - 1.0) <= 1e-5
        for value in values
    )
    if len(active) != 1 or not inactive_ok:
        raise ValueError(f"{label} must be a one-hot vector")
    return active[0]


def validate_rule_features(
    features: Sequence[float],
    ruleset: RulesetSpec | None = None,
    label: str = "rule_features",
) -> None:
    if len(features) != 10:
        raise ValueError(f"{label} must have length 10")
    scoring = _active_one_hot_index(features[0:4], f"{label}[0:4]")
    ko = _active_one_hot_index(features[4:6], f"{label}[4:6]")
    suicide = _active_one_hot_index(features[6:8], f"{label}[6:8]")
    for index in (8, 9):
        value = float(features[index])
        if value < -1.0 or value > 1.0:
            raise ValueError(f"{label}[{index}] must be in [-1, 1]")
    if ruleset is None:
        return
    expected = (
        SCORING_INDEX[ruleset.scoring],
        KO_INDEX[ruleset.ko],
        SUICIDE_INDEX[ruleset.suicide],
    )
    actual = (scoring, ko, suicide)
    if actual != expected:
        raise ValueError(
            f"{label} one-hots do not match ruleset {ruleset.name!r}: "
            f"expected {expected}, got {actual}"
        )
