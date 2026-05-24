from seismonn.data.download import ensure_data_available


def test_ensure_data_available_does_nothing_when_files_exist(tmp_path):
    metadata_path = tmp_path / "data" / "metadata.csv"
    data_dir = tmp_path / "2nd_selection"

    metadata_path.parent.mkdir(parents=True)
    data_dir.mkdir()

    metadata_path.write_text("sample_id,path\n", encoding="utf-8")

    ensure_data_available(
        metadata_path="data/metadata.csv",
        data_dir="2nd_selection",
        repo_root=tmp_path,
        use_dvc=False,
        allow_huggingface_fallback=False,
    )
