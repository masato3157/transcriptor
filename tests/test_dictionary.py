import json

from dictionary import load_dictionary, save_dictionary, apply_replacements


def test_load_dictionary_returns_empty_dict_when_file_missing(tmp_path):
    result = load_dictionary(tmp_path / "does_not_exist.json")
    assert result == {}


def test_save_and_load_dictionary_round_trip(tmp_path):
    path = tmp_path / "dictionary.json"
    mapping = {"おおたかゆうこ": "大高ゆうこ", "どりーむわーく": "ドリームワーク"}

    save_dictionary(mapping, path)
    result = load_dictionary(path)

    assert result == mapping


def test_save_dictionary_writes_readable_utf8_json(tmp_path):
    path = tmp_path / "dictionary.json"
    save_dictionary({"おおたかゆうこ": "大高ゆうこ"}, path)

    content = path.read_text(encoding="utf-8")
    assert "大高ゆうこ" in content  # not escaped as \uXXXX
    assert json.loads(content) == {"おおたかゆうこ": "大高ゆうこ"}


def test_apply_replacements_replaces_all_matches():
    text = "おおたかゆうこです。おおたかゆうこと申します。"
    mapping = {"おおたかゆうこ": "大高ゆうこ"}

    result = apply_replacements(text, mapping)

    assert result == "大高ゆうこです。大高ゆうこと申します。"


def test_apply_replacements_with_empty_mapping_returns_text_unchanged():
    assert apply_replacements("そのままです。", {}) == "そのままです。"


def test_apply_replacements_with_multiple_entries_applies_all():
    text = "おおたかゆうこがどりーむわーくについて話します。"
    mapping = {"おおたかゆうこ": "大高ゆうこ", "どりーむわーく": "ドリームワーク"}

    result = apply_replacements(text, mapping)

    assert result == "大高ゆうこがドリームワークについて話します。"
