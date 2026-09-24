"""Guard the official 447-class classifier order used by cached predictions."""

from waves_sed.labels import load_labels


def test_labels_have_complete_unique_audioset_strong_classes():
    class_ids, class_names = load_labels()

    assert isinstance(class_ids, tuple)
    assert isinstance(class_names, tuple)
    assert len(class_ids) == len(class_names) == 447
    assert len(set(class_ids)) == 447
    assert all(class_id.startswith(("/m/", "/g/", "/t/")) for class_id in class_ids)
    assert all(isinstance(name, str) and name for name in class_names)


def test_labels_preserve_checkpoint_order_instead_of_csv_id_order():
    class_ids, class_names = load_labels()

    assert class_names[0] == "Accelerating, revving, vroom"
    assert class_ids[0] == "/m/07q2z82"
    assert (class_ids[-1], class_names[-1]) == ("/m/01s0vc", "Zipper (clothing)")
    assert class_names == tuple(sorted(class_names))
    assert class_ids != tuple(sorted(class_ids))
