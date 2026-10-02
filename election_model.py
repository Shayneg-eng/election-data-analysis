import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── Columns to always exclude ─────────────────────────────────────────────────
# Outcome columns (leakage) and free-text columns (not usable as features).
OUTCOMES = {
    'won_election', 'popular_vote_total', 'popular_vote_share_pct',
    'electoral_votes_received', 'victory_margin_popular_pct',
}
FREE_TEXT = {
    'notes', 'first_of_demographic_group_description', 'political_dynasty_relation',
    'russia_aggression_description', 'china_aggression_description',
    'major_us_terror_attack_description', 'major_financial_crisis_description',
    'major_gun_legislation_description', 'landmark_scotus_decisions_list',
    'major_nuclear_accident_description', 'pre_covid_who_pandemic_description',
    'major_political_assassination_description', 'us_combat_theater',
    'major_us_embassy_attack_description', 'foreign_cyberattack_description',
    'nato_new_members', 'presidential_impeachment_description',
    # Institution name strings — too many unique values to label-encode meaningfully
    'education_undergraduate_institution', 'education_graduate_institution',
    # Name strings
    'vp_candidate_name', 'third_party_candidate_name',
}
EXCLUDE = OUTCOMES | FREE_TEXT

# ── Tier 1 — SAFE ─────────────────────────────────────────────────────────────
# Pre-campaign economic fundamentals + candidate biography + factual political
# context.  Everything here can be verified from official records / statistics.
SAFE = [
    # Economic conditions (shared per election year)
    'gdp_growth_rate_election_year', 'gdp_growth_rate_q2_q3',
    'unemployment_rate_october', 'unemployment_rate_change_24mo',
    'cpi_inflation_rate_annual', 'cpi_inflation_rate_change_yoy',
    'real_wage_growth_rate', 'sp500_return_ytd', 'sp500_return_24mo',
    'federal_deficit_pct_gdp', 'national_debt_pct_gdp',
    'recession_flag', 'recession_months_in_prior_12',
    'oil_price_election_month', 'oil_price_change_12mo', 'oil_embargo_or_shock_flag',
    'housing_price_index_change_yoy', 'consumer_confidence_index',
    'manufacturing_jobs_change_12mo', 'trade_deficit_pct_gdp',
    'federal_funds_rate_election_day', 'federal_funds_rate_change_24mo',
    'bank_failures_count_4yr', 'foreclosure_rate', 'poverty_rate',
    'gini_coefficient', 'student_loan_total_trillions', 'opioid_overdose_deaths_annual',
    'us_life_expectancy_change_4yr', 'labor_force_participation_rate', 'union_membership_rate',
    # Military / geopolitical events
    'us_troops_in_combat_flag', 'us_combat_deaths_election_year',
    'us_combat_deaths_total_conflict', 'us_declared_war_flag',
    'nato_article5_invoked_flag', 'russia_military_aggression_flag',
    'china_military_aggression_flag', 'china_tariffs_on_us_flag', 'us_tariffs_on_china_flag',
    'major_us_terror_attack_flag', 'us_terror_attack_deaths_4yr',
    'iran_nuclear_agreement_status', 'north_korea_nuclear_test_flag',
    'north_korea_nuclear_test_count', 'global_oil_cartel_production_cut_flag',
    'un_security_council_us_vetoes_4yr',
    'soviet_russian_nuclear_tests_4yr', 'china_gdp_growth_rate',
    'china_trade_surplus_with_us_billions', 'major_us_embassy_attack_flag',
    'foreign_cyberattack_flag', 'russian_chinese_nuclear_doctrine_change_flag',
    'us_ambassador_expelled_count', 'major_allied_leadership_change_flag',
    'nato_enlargement_flag', 'eu_refugee_crisis_flag', 'us_foreign_aid_pct_gdp',
    'allied_combat_deaths_with_us', 'imf_emergency_lending_flag',
    'fao_food_price_index_yoy_change', 'major_nuclear_accident_flag',
    'pre_covid_who_pandemic_flag', 'canal_disruption_flag',
    # Domestic political events
    'presidential_impeachment_flag', 'major_financial_crisis_flag',
    'declared_federal_disasters_4yr', 'pandemic_flag', 'pandemic_us_deaths',
    'supreme_court_vacancies_filled_term', 'federal_minimum_wage_change_flag',
    'roe_v_wade_status', 'major_gun_legislation_flag',
    'federal_debt_ceiling_crisis_flag', 'government_shutdown_count_4yr',
    'government_shutdown_total_days_4yr', 'crime_rate_change_4yr',
    'illegal_border_crossings_change_4yr', 'us_uninsured_rate',
    'presidential_assassination_attempt_flag', 'major_political_assassination_flag',
    'mass_casualty_shooting_events_4yr', 'mass_casualty_shooting_deaths_4yr',
    'national_guard_federally_activated_flag', 'major_scotus_decisions_count_4yr',
    'federal_execution_count_4yr', 'federal_schedule1_legalization_flag',
    'major_infrastructure_failure_flag', 'major_tech_regulation_flag',
    # Candidate biography
    'candidate_age_election_day', 'candidate_gender', 'candidate_race_ethnicity',
    'candidate_religion', 'candidate_birth_state', 'candidate_home_state',
    'education_highest_degree', 'military_service_flag', 'military_branch',
    'military_highest_rank', 'military_combat_service_flag', 'military_combat_decorations_flag',
    'years_total_elected_office', 'years_federal_elected_office', 'years_as_governor',
    'years_in_us_senate', 'years_in_us_house', 'years_as_vp', 'years_in_cabinet',
    'prior_presidential_campaigns', 'incumbent_president_flag', 'incumbent_vp_flag',
    'first_of_demographic_group_flag', 'political_dynasty_flag',
    'net_worth_fec_disclosure_millions',
    # Incumbent record (if applicable)
    'bills_signed_into_law_term', 'executive_orders_issued_term',
    'federal_civilian_employment_change_term', 'state_visits_count_term',
    'treaties_ratified_term', 'presidential_veto_count_term', 'presidential_veto_overrides_term',
    # Third-party context
    'third_party_popular_vote_share',
]

# ── Tier 2 — SAFE + MODERATE ──────────────────────────────────────────────────
# Adds campaign-era measurable data: speech corpus statistics (computed from
# transcripts), debate metrics, campaign spending, endorsements, primary results,
# VP background, and social media counts.
MODERATE = SAFE + [
    # Speech corpus statistics
    'speech_corpus_total_words', 'flesch_kincaid_grade_level', 'flesch_reading_ease',
    'avg_words_per_sentence', 'avg_syllables_per_word', 'type_token_ratio',
    'first_person_singular_pct', 'first_person_plural_pct', 'second_person_pct',
    'negative_emotion_word_pct', 'positive_emotion_word_pct', 'anxiety_word_pct',
    'anger_word_pct', 'certainty_word_pct', 'tentative_word_pct', 'religion_word_pct',
    'money_word_pct', 'power_word_pct', 'future_tense_word_pct', 'past_tense_word_pct',
    'opponent_name_mention_count', 'opponent_name_mention_rate',
    'question_mark_count', 'exclamation_mark_count',
    # Debate activity
    'debate_words_spoken_total', 'debate_interruption_count',
    # Campaign spending & activity
    'campaign_total_spending_millions', 'campaign_spending_advantage',
    'general_election_spending_millions', 'public_financing_accepted_flag',
    'campaign_ad_spend_millions', 'states_visited_general_election',
    'total_campaign_events_general', 'debate_appearances_general',
    'debate_appearances_skipped', 'vp_debate_count',
    # Endorsements
    'major_newspaper_endorsements', 'major_newspaper_opponent_endorsements',
    'sitting_president_endorsed_flag', 'former_presidents_same_party_endorsement_count',
    'senate_same_party_endorsement_count', 'governor_same_party_endorsement_count',
    # Primary performance
    'primary_vote_share_pct', 'primary_opponents_count',
    # VP profile
    'vp_home_state', 'vp_years_political_experience',
    'vp_prior_presidential_campaigns', 'vp_military_service_flag', 'convention_held_first',
    # Social media (modern cycles only — NaN for earlier years)
    'social_media_followers_election_day', 'social_media_posts_90days',
]

# ── Tier 3 — ALL ──────────────────────────────────────────────────────────────
# Adds opponent-comparison deltas computed from both candidates' records.
ALL_FEATURES = MODERATE + [
    'age_difference', 'experience_years_difference', 'spending_ratio',
    'endorsement_count_difference', 'debate_words_difference',
    'primary_opponent_count_difference',
]

TIERS = {
    'SAFE only':       SAFE,
    'SAFE + MODERATE': MODERATE,
    'ALL':             ALL_FEATURES,
}

# ── Load & prepare ────────────────────────────────────────────────────────────
raw = pd.read_csv(
    'us_presidential_elections_1960_2024.csv',
    quotechar='"',
    na_values=['NULL', 'null', 'N/A', 'n/a', '', ' '],
    keep_default_na=True,
)
raw['target'] = (raw['won_election'].astype(str).str.upper() == 'TRUE').astype(int)
# Catch any remaining NULL-like strings not caught by na_values
raw.replace(r'^\s*$', np.nan, regex=True, inplace=True)
# Drop rows where candidate_name is missing (malformed third-party rows)
raw = raw[raw['candidate_name'].notna()].reset_index(drop=True)

META       = ['candidate_name', 'election_year', 'party', 'target']
elections  = sorted(raw['election_year'].unique())


def prepare_X(df, feature_cols):
    """Encode and impute a feature matrix."""
    X = df[feature_cols].copy()
    # Normalise any residual NULL-like strings to NaN
    X.replace(r'^\s*(NULL|null|N/A|n/a|nan)\s*$', np.nan, regex=True, inplace=True)
    for col in X.select_dtypes(include='bool').columns:
        X[col] = X[col].astype(int)
    for col in X.select_dtypes(include='object').columns:
        str_vals = X[col].dropna().astype(str)
        if not str_vals.empty and set(str_vals.str.upper().unique()) <= {'TRUE', 'FALSE'}:
            X[col] = (X[col].astype(str).str.upper() == 'TRUE').astype(int)
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].fillna('MISSING').astype(str))
    # Impute numeric NaNs with column median; fallback to 0 for all-NaN columns
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X


DEFAULT_PARAMS = dict(
    n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8,
    eval_metric='logloss', random_state=42, verbosity=0,
)


def run_loeo(df, feature_cols, model_params=None):
    """Leave-one-election-out CV. Returns per-candidate results DataFrame."""
    if model_params is None:
        model_params = DEFAULT_PARAMS
    X_all = prepare_X(df, feature_cols)
    y_all = df['target']
    groups = df['election_year']
    results = []

    for test_year in elections:
        train_mask = (groups != test_year).values
        test_mask  = (groups == test_year).values
        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train         = y_all[train_mask]

        model = XGBClassifier(**model_params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        for name, party, actual, prob in zip(
            df.loc[test_mask, 'candidate_name'].values,
            df.loc[test_mask, 'party'].values,
            y_all[test_mask].values,
            proba,
        ):
            results.append({
                'year': test_year, 'candidate': name, 'party': party,
                'actual_winner': bool(actual), 'win_probability': round(prob, 3),
            })

    res = pd.DataFrame(results)
    res['predicted_winner'] = False
    for year, grp in res.groupby('year'):
        res.loc[grp['win_probability'].idxmax(), 'predicted_winner'] = True
    res['correct'] = res['actual_winner'] == res['predicted_winner']
    return res


def election_accuracy(res):
    """Fraction of elections where the highest-probability candidate actually won."""
    correct = sum(
        res.loc[grp['win_probability'].idxmax(), 'actual_winner']
        for _, grp in res.groupby('year')
    )
    return correct / len(elections)


# ── Hyperparameter grid search (using SAFE+MODERATE tier) ─────────────────────
import itertools

SEARCH_TIER = 'SAFE + MODERATE'
search_cols = [c for c in MODERATE if c in raw.columns and c not in EXCLUDE]

param_grid = {
    'n_estimators':    [50, 100, 200, 400],
    'max_depth':       [2, 3, 4],
    'learning_rate':   [0.01, 0.05, 0.1],
    'subsample':       [0.6, 0.8, 1.0],
    'colsample_bytree':[0.5, 0.7, 1.0],
    'reg_alpha':       [0, 0.5, 2.0],
    'reg_lambda':      [1, 5, 10],
    'min_child_weight':[1, 3, 5],
}

print("Searching hyperparameters (this may take a moment)...", flush=True)

# Random search: sample combinations rather than exhaustive grid
import random
random.seed(42)

def random_params():
    return {k: random.choice(v) for k, v in param_grid.items()} | {
        'eval_metric': 'logloss', 'random_state': 42, 'verbosity': 0,
    }

best_acc    = -1
best_params = None
n_trials    = 200

for _ in range(n_trials):
    p   = random_params()
    res = run_loeo(raw, search_cols, model_params=p)
    acc = election_accuracy(res)
    if acc > best_acc:
        best_acc    = acc
        best_params = p

print(f"Best election accuracy on {SEARCH_TIER}: {best_acc:.1%}")
print("Best params:")
for k, v in best_params.items():
    if k not in ('eval_metric', 'random_state', 'verbosity'):
        print(f"  {k:<22} = {v}")
print()

# ── Run all three tiers and collect summary ───────────────────────────────────
tier_results = {}
for tier_name, feat_cols in TIERS.items():
    # keep only columns that actually exist in the data and aren't excluded
    valid_cols = [c for c in feat_cols if c in raw.columns and c not in EXCLUDE]
    tier_results[tier_name] = run_loeo(raw, valid_cols, model_params=best_params)

# ── Per-election results table ────────────────────────────────────────────────
# Build a unified view: one row per (year, candidate) showing all three tiers
first_tier_df = tier_results['SAFE only'][['year', 'candidate', 'party', 'actual_winner']].copy()
for tier_name, res in tier_results.items():
    short = tier_name.split()[0]  # SAFE / SAFE+MODERATE → first word
    label = {'SAFE': 'safe', 'SAFE': 'safe', 'ALL': 'all'}.get(short, short.lower())
    label = 'safe' if tier_name == 'SAFE only' else ('mod' if tier_name == 'SAFE + MODERATE' else 'all')
    first_tier_df = first_tier_df.merge(
        res[['year', 'candidate', 'win_probability', 'predicted_winner', 'correct']]
          .rename(columns={
              'win_probability':  f'prob_{label}',
              'predicted_winner': f'pred_{label}',
              'correct':          f'ok_{label}',
          }),
        on=['year', 'candidate'],
    )

print("=" * 95)
print("  LEAVE-ONE-ELECTION-OUT: PER-CANDIDATE RESULTS ACROSS ALL THREE FEATURE TIERS")
print("=" * 95)
hdr = (f"{'Year':<6} {'Candidate':<25} {'Actual':>6}  "
       f"{'──── SAFE ────':^17}  {'── SAFE+MOD ──':^17}  {'──── ALL ────':^17}")
print(hdr)
print(f"{'':38}  {'Prob':>6} {'Pred':>5} {'✓':>2}  "
      f"{'Prob':>6} {'Pred':>5} {'✓':>2}  "
      f"{'Prob':>6} {'Pred':>5} {'✓':>2}")
print("-" * 95)

for _, row in first_tier_df.iterrows():
    actual_str = 'WIN ' if row['actual_winner'] else 'LOSS'
    def fmt(prob, pred, ok):
        return f"{prob:>6.3f} {'WIN':>5}" if pred else f"{prob:>6.3f} {'LOSS':>5}", '✓' if ok else '✗'
    s_str, s_ok = fmt(row['prob_safe'], row['pred_safe'], row['ok_safe'])
    m_str, m_ok = fmt(row['prob_mod'],  row['pred_mod'],  row['ok_mod'])
    a_str, a_ok = fmt(row['prob_all'],  row['pred_all'],  row['ok_all'])
    print(f"{row['year']:<6} {row['candidate']:<25} {actual_str:>6}  "
          f"{s_str}  {s_ok}  {m_str}  {m_ok}  {a_str}  {a_ok}")

# ── Summary comparison ────────────────────────────────────────────────────────
print()
print("=" * 75)
print("  ACCURACY COMPARISON")
print("=" * 75)
print(f"  {'Tier':<26}   {'Cand. correct':>14}   {'Cand. acc':>10}   "
      f"{'Elections':>10}   {'Election acc':>12}")
print(f"  {'-'*26}   {'-'*14}   {'-'*10}   {'-'*10}   {'-'*12}")

for tier_name, res in tier_results.items():
    n_cands    = len(res)
    cand_ok    = res['correct'].sum()
    cand_acc   = cand_ok / n_cands
    elec_ok    = sum(
        res.loc[grp['win_probability'].idxmax(), 'actual_winner']
        for _, grp in res.groupby('year')
    )
    elec_acc   = elec_ok / len(elections)

    n_feat = len([c for c in TIERS[tier_name] if c in raw.columns and c not in EXCLUDE])
    label  = f"{tier_name} ({n_feat} feats)"
    print(f"  {label:<35}   {cand_ok}/{n_cands}   {cand_acc:>10.1%}   "
          f"{elec_ok}/{len(elections)}   {elec_acc:>12.1%}")

# ── Highlight elections where tier changes the call ──────────────────────────
print()
print("=" * 75)
print("  ELECTIONS WHERE TIER CHANGES THE PREDICTED WINNER")
print("=" * 75)
print(f"  {'Year':<6} {'Actual winner':<25} {'SAFE':>15} {'SAFE+MOD':>15} {'ALL':>15}")
print(f"  {'-'*6} {'-'*25} {'-'*15} {'-'*15} {'-'*15}")

for year in elections:
    actual = first_tier_df.loc[
        first_tier_df['year'] == year, ['candidate', 'actual_winner']
    ]
    winner_name = actual.loc[actual['actual_winner'], 'candidate'].values
    winner_name = winner_name[0] if len(winner_name) else '?'

    preds = {}
    for col, label in [('pred_safe', 'SAFE'), ('pred_mod', 'SAFE+MOD'), ('pred_all', 'ALL')]:
        row_pred = first_tier_df.loc[
            (first_tier_df['year'] == year) & first_tier_df[col], 'candidate'
        ]
        preds[label] = row_pred.values[0] if len(row_pred) else '?'

    if len(set(preds.values())) > 1:
        def mark(pred):
            return f"{'✓' if pred == winner_name else '✗'} {pred}"
        print(f"  {year:<6} {winner_name:<25} {mark(preds['SAFE']):>15} "
              f"{mark(preds['SAFE+MOD']):>15} {mark(preds['ALL']):>15}")

print()
