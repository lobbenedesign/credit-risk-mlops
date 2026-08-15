from creditrisk.data.synthetic import FEATURE_COLUMNS, generate_synthetic_applications


class TestGenerateSyntheticApplications:
    def test_generator_is_deterministic(self):
        a = generate_synthetic_applications(n=500, seed=7)
        b = generate_synthetic_applications(n=500, seed=7)
        assert a.applications["application_id"].tolist() == b.applications["application_id"].tolist()
        assert a.applications["defaulted"].tolist() == b.applications["defaulted"].tolist()

    def test_produces_the_requested_number_of_rows(self):
        ds = generate_synthetic_applications(n=250, seed=1)
        assert len(ds.applications) == 250

    def test_all_feature_columns_are_present(self):
        ds = generate_synthetic_applications(n=100, seed=1)
        for col in FEATURE_COLUMNS:
            assert col in ds.applications.columns

    def test_protected_group_is_binary_and_imbalanced_by_design(self):
        ds = generate_synthetic_applications(n=3000, seed=42)
        counts = ds.applications["protected_group"].value_counts(normalize=True)
        assert set(counts.index) == {"A", "B"}
        assert counts["A"] > counts["B"]  # A is the majority group, ~70/30 by construction

    def test_default_rate_is_realistic_not_near_50_percent(self):
        # regression guard for the calibration described in docs/adr/0002:
        # an earlier version of the generator produced a ~50% default rate
        ds = generate_synthetic_applications(n=3000, seed=42)
        rate = ds.applications["defaulted"].mean()
        assert 0.10 < rate < 0.35

    def test_true_default_rate_is_close_across_protected_groups(self):
        # fair by construction: protected_group has no causal effect on
        # `defaulted` in the generator — any gap here is sampling noise
        ds = generate_synthetic_applications(n=3000, seed=42)
        rates = ds.applications.groupby("protected_group")["defaulted"].mean()
        assert abs(rates["A"] - rates["B"]) < 0.05

    def test_credit_history_score_is_systematically_lower_for_group_b(self):
        # the deliberate proxy-bias signal the fairness report is meant to
        # catch — see the module docstring and docs/adr/0002
        ds = generate_synthetic_applications(n=3000, seed=42)
        means = ds.applications.groupby("protected_group")["credit_history_score"].mean()
        assert means["A"] - means["B"] > 40

    def test_credit_history_score_stays_within_bureau_range(self):
        ds = generate_synthetic_applications(n=1000, seed=1)
        assert ds.applications["credit_history_score"].between(300, 850).all()

    def test_applications_are_sorted_by_date(self):
        ds = generate_synthetic_applications(n=500, seed=1)
        dates = ds.applications["application_date"].tolist()
        assert dates == sorted(dates)
