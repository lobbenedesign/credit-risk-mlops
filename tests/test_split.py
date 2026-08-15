import pytest

from creditrisk.data.synthetic import generate_synthetic_applications
from creditrisk.model.split import split_out_of_time


class TestSplitOutOfTime:
    def test_train_and_test_do_not_overlap_and_cover_everything(self):
        ds = generate_synthetic_applications(n=1000, seed=1)
        split = split_out_of_time(ds.applications, test_fraction=0.25)
        train_ids = set(split.train_df["application_id"])
        test_ids = set(split.test_df["application_id"])
        assert train_ids.isdisjoint(test_ids)
        assert len(train_ids) + len(test_ids) == len(ds.applications)

    def test_every_train_row_predates_every_test_row(self):
        ds = generate_synthetic_applications(n=1000, seed=1)
        split = split_out_of_time(ds.applications, test_fraction=0.25)
        assert split.train_df["application_date"].max() <= split.test_df["application_date"].min()

    def test_test_fraction_controls_the_split_size_approximately(self):
        ds = generate_synthetic_applications(n=1000, seed=1)
        split = split_out_of_time(ds.applications, test_fraction=0.25)
        assert abs(len(split.test_df) / len(ds.applications) - 0.25) < 0.01

    def test_invalid_test_fraction_raises(self):
        ds = generate_synthetic_applications(n=100, seed=1)
        with pytest.raises(ValueError):
            split_out_of_time(ds.applications, test_fraction=0.0)
        with pytest.raises(ValueError):
            split_out_of_time(ds.applications, test_fraction=1.0)
