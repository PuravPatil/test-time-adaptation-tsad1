"""Convert data/raw/<dataset>/ into the common format in data/processed/<dataset>/.

Common format per dataset: train.npy (T_train, D), test.npy (T_test, D),
test_labels.npy (T_test,) with 0 = normal, 1 = anomaly.

TODO (weeks 1-3): one function per dataset (smd, msl, smap, creditcard).
"""


def main() -> None:
    raise NotImplementedError("Preprocessing not written yet")


if __name__ == "__main__":
    main()
