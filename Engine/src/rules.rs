#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringRule {
    Area,
    AreaAncientChinese,
    Territory,
    TerritoryWithSekiScore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KoRule {
    SimpleKo,
    PositionalSuperKo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuicideRule {
    Allowed,
    Forbidden,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ruleset {
    pub scoring: ScoringRule,
    pub ko: KoRule,
    pub suicide: SuicideRule,
    pub komi: f32,
}

impl Ruleset {
    pub const fn new(scoring: ScoringRule, ko: KoRule, suicide: SuicideRule, komi: f32) -> Self {
        Self {
            scoring,
            ko,
            suicide,
            komi,
        }
    }
}

impl Default for Ruleset {
    fn default() -> Self {
        Self {
            scoring: ScoringRule::Area,
            ko: KoRule::SimpleKo,
            suicide: SuicideRule::Forbidden,
            komi: 7.5,
        }
    }
}
