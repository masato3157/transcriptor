from pathlib import Path

import pytest

from exporter import (
    format_timestamp_txt,
    format_timestamp_srt,
    format_timestamp_vtt,
    build_txt,
    build_srt,
    build_vtt,
    export,
)


SEGMENTS_WITH_SPEAKERS = [
    {"start": 5.0, "end": 7.0, "speaker": "SPEAKER_00", "text": "おはようございます。"},
    {"start": 8.0, "end": 10.0, "speaker": "SPEAKER_01", "text": "よろしくお願いします。"},
]

SEGMENTS_NO_SPEAKERS = [
    {"start": 5.0, "end": 7.0, "speaker": None, "text": "おはようございます。"},
]

SPEAKER_NAMES = {"SPEAKER_00": "田中", "SPEAKER_01": "鈴木"}


def test_format_timestamp_txt():
    assert format_timestamp_txt(5.0) == "00:00:05"
    assert format_timestamp_txt(3665.0) == "01:01:05"


def test_format_timestamp_srt():
    assert format_timestamp_srt(5.123) == "00:00:05,123"


def test_format_timestamp_vtt():
    assert format_timestamp_vtt(5.123) == "00:00:05.123"


def test_build_txt_with_timestamps_and_speakers():
    result = build_txt(SEGMENTS_WITH_SPEAKERS, SPEAKER_NAMES, include_timestamps=True)
    assert result == (
        "[00:00:05] 田中: おはようございます。\n"
        "[00:00:08] 鈴木: よろしくお願いします。\n"
    )


def test_build_txt_without_timestamps():
    result = build_txt(SEGMENTS_WITH_SPEAKERS, SPEAKER_NAMES, include_timestamps=False)
    assert result == (
        "田中: おはようございます。\n"
        "鈴木: よろしくお願いします。\n"
    )


def test_build_txt_without_speakers():
    result = build_txt(SEGMENTS_NO_SPEAKERS, {}, include_timestamps=False)
    assert result == "おはようございます。\n"


def test_build_txt_falls_back_to_label_when_name_missing():
    result = build_txt(SEGMENTS_WITH_SPEAKERS, {}, include_timestamps=False)
    assert result == (
        "SPEAKER_00: おはようございます。\n"
        "SPEAKER_01: よろしくお願いします。\n"
    )


def test_build_srt_format():
    result = build_srt(SEGMENTS_WITH_SPEAKERS, SPEAKER_NAMES)
    assert result == (
        "1\n00:00:05,000 --> 00:00:07,000\n田中: おはようございます。\n\n"
        "2\n00:00:08,000 --> 00:00:10,000\n鈴木: よろしくお願いします。\n"
    )


def _cue_texts(srt):
    return [block.split("\n", 2)[2] for block in srt.strip("\n").split("\n\n")]


def test_build_srt_starts_a_new_cue_at_each_sentence_end():
    # Regression (V07): a cue used to be packed up to the width limit, so the
    # start of the next sentence ("順番に") got glued to the end of the
    # previous cue. A sentence end (。) must always end the cue. Timing is the
    # segment's duration split by character count: 17 of 31 chars -> 1.7s.
    text = "今日は2つのご質問が届いています。順番にお答えしていきますね。"
    segments = [{"start": 0.0, "end": 3.1, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    assert result == (
        "1\n00:00:00,000 --> 00:00:01,700\n今日は2つのご質問が届いています。\n\n"
        "2\n00:00:01,700 --> 00:00:03,100\n順番にお答えしていきますね。\n"
    )


def test_build_srt_treats_a_space_between_sentences_as_a_boundary():
    # Whisper often separates sentences with a space instead of "。", which
    # used to leave "夢" at the end of one cue and "主さんは…" in the next.
    text = "その中には予知無駄としか思えない事例もあります 夢主さんは60代の女性です"
    segments = [{"start": 0.0, "end": 6.0, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    assert _cue_texts(result) == [
        "その中には予知無駄としか思えない事例もあります",
        "夢主さんは60代の女性です",
    ]


def test_build_srt_does_not_treat_a_space_inside_latin_text_as_a_boundary():
    segments = [{"start": 0.0, "end": 2.0, "speaker": None, "text": "iPhone 15を使っています"}]

    result = build_srt(segments, {})

    assert _cue_texts(result) == ["iPhone 15を使っています"]


def test_build_srt_ends_a_sentence_after_a_sentence_final_particle():
    # No punctuation and no space: "ね" (終助詞) followed by a new clause.
    text = "結論から言いますね私は予知無を否定はしません"
    segments = [{"start": 0.0, "end": 4.0, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    assert _cue_texts(result) == ["結論から言いますね", "私は予知無を否定はしません"]


def test_build_srt_ends_a_sentence_after_polite_ます_followed_by_a_new_clause():
    text = "よかったらチャンネル登録お願いしますまた概要欄をご覧ください"
    segments = [{"start": 0.0, "end": 5.0, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    assert _cue_texts(result) == [
        "よかったらチャンネル登録お願いします",
        "また概要欄をご覧ください",
    ]


def test_build_srt_ends_a_sentence_after_polite_past_でした_followed_by_a_new_clause():
    text = "何も思い当たることはありませんでした私は驚きました"
    segments = [{"start": 0.0, "end": 4.0, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    assert _cue_texts(result) == ["何も思い当たることはありませんでした", "私は驚きました"]


def test_build_srt_keeps_a_sentence_that_fits_on_one_line_as_one_cue():
    # Exactly the V07 sentence that used to be cut at "リスナーから|のお便り":
    # 24 characters, so it must not be split at all.
    text = "今回はリスナーからのお便り、ご質問を紹介します。"
    segments = [{"start": 0.0, "end": 4.0, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    assert _cue_texts(result) == [text]


def test_build_srt_splits_a_long_sentence_at_a_comma_when_that_balances_the_lines():
    text = "夢の中で見た景色がとても鮮やかで、目が覚めたあともずっと忘れられませんでした"
    segments = [{"start": 0.0, "end": 7.6, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    assert _cue_texts(result) == [
        "夢の中で見た景色がとても鮮やかで、",
        "目が覚めたあともずっと忘れられませんでした",
    ]


def test_build_srt_splits_a_long_sentence_at_the_reason_clause_not_after_は():
    # The break is after "規則で" (the reason), keeping "これ以上のことは" with
    # its predicate, rather than after the topic marker "は".
    text = "ですが規則でこれ以上のことは申し上げられないんです。"
    segments = [{"start": 0.0, "end": 5.2, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    assert _cue_texts(result) == [
        "ですが規則で",
        "これ以上のことは申し上げられないんです。",
    ]


def test_build_srt_ends_a_sentence_after_polite_ましょう_followed_by_a_new_clause():
    text = "では見ていきましょう次の夢です"
    segments = [{"start": 0.0, "end": 3.0, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    assert _cue_texts(result) == ["では見ていきましょう", "次の夢です"]


def test_build_srt_ends_a_sentence_after_ください_followed_by_a_new_clause():
    text = "概要欄をご覧くださいまた次回お会いしましょう"
    segments = [{"start": 0.0, "end": 4.0, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    assert _cue_texts(result) == ["概要欄をご覧ください", "また次回お会いしましょう"]


def test_build_srt_starts_a_new_phrase_after_a_comma():
    # Regression (V07): Janome tags the "後" of "、後から" as a suffix, which
    # glued "後から" onto the phrase before the comma and made the comma
    # unusable as a break point (the line was cut after "後から" instead).
    text = "夢を見た日を記録していたおかげで、後からこの一致に気づけたそうです。"
    segments = [{"start": 0.0, "end": 3.4, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    assert _cue_texts(result) == [
        "夢を見た日を記録していたおかげで、",
        "後からこの一致に気づけたそうです。",
    ]


def test_build_srt_does_not_cut_a_honorific_verb_phrase_in_two():
    # Regression (V07): "おやりに|なっている" was cut between the particle and
    # its verb. Breaking after the whole modifier clause is preferred.
    text = "夢占いと大鷹さんがおやりになっている大高式ドリームワークは何が違うのですか"
    segments = [{"start": 0.0, "end": 3.7, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    assert _cue_texts(result) == [
        "夢占いと大鷹さんがおやりになっている",
        "大高式ドリームワークは何が違うのですか",
    ]


def test_build_srt_avoids_leaving_a_very_short_last_line():
    # Regression (V07): the sentence used to end with a lone "分野です".
    text = "私が学んだのはトランスパーソナル心理学という分野です"
    segments = [{"start": 0.0, "end": 5.2, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    assert _cue_texts(result) == [
        "私が学んだのは",
        "トランスパーソナル心理学という分野です",
    ]


def test_build_srt_does_not_break_right_after_の():
    # Regression (V07): "夢の|伝える意味は…" left the possessive/subject の
    # dangling at the end of a line.
    text = "同じ黒いスーツでも夢の伝える意味はその人によって違います"
    segments = [{"start": 0.0, "end": 5.6, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    assert _cue_texts(result) == [
        "同じ黒いスーツでも夢の伝える意味は",
        "その人によって違います",
    ]


def test_build_srt_splits_an_oversized_word_into_even_pieces():
    # A single unbroken token longer than the width (a run of latin letters
    # that Janome treats as one noun) has no word boundary to break on, so
    # it is cut into equal pieces rather than 25 + a short remainder.
    segments = [{"start": 0.0, "end": 3.0, "speaker": None, "text": "A" * 30}]

    result = build_srt(segments, {})

    assert result == (
        "1\n00:00:00,000 --> 00:00:01,500\n" + "A" * 15 + "\n\n"
        "2\n00:00:01,500 --> 00:00:03,000\n" + "A" * 15 + "\n"
    )


def test_build_srt_never_leaves_punctuation_at_the_start_of_a_line():
    # Regression: breaking right after a particle ("は" in "時は") that is
    # immediately followed by a "、" in the source text used to strand that
    # "、" as the first character of the next line. Punctuation must stay
    # attached to the end of the line it follows.
    text = "ですから、夢を記録する時は、目覚めてすぐ、起き上がる前にメモすることが大切です。"
    segments = [{"start": 0.0, "end": 8.32, "speaker": None, "text": text}]

    result = build_srt(segments, {})

    for line in result.split("\n"):
        if not line or "-->" in line or line.isdigit():
            continue
        assert line[0] not in "。、！？", f"line starts with punctuation: {line!r}"


def test_build_srt_never_leaves_punctuation_at_the_start_of_a_segment():
    # Regression: Whisper sometimes splits its OWN segments such that a
    # comma/period lands at the very start of the next segment's raw text
    # (before any line-wrapping happens), rather than at the end of the
    # previous segment. This is a different case from the within-one-wrap
    # bug above: here segment 2's own text begins with "、".
    segments = [
        {"start": 19.610, "end": 22.106, "speaker": None, "text": "ですから、夢を記録する時は"},
        {
            "start": 22.106,
            "end": 27.290,
            "speaker": None,
            "text": "、目覚めてすぐ、起き上がる前にメモすることが大切です。",
        },
    ]

    result = build_srt(segments, {})

    for line in result.split("\n"):
        if not line or "-->" in line or line.isdigit():
            continue
        assert line[0] not in "。、！？", f"line starts with punctuation: {line!r}"
    # The leading "、" should have been reattached to the end of segment 1's text.
    assert "ですから、夢を記録する時は、" in result


def test_build_srt_drops_a_segment_that_is_only_punctuation():
    # Whisper occasionally emits a whole segment whose text is a single
    # punctuation mark (e.g. a lone "。" a fraction of a second long). Once
    # that mark is pulled onto the previous segment, the now-empty segment
    # should disappear entirely rather than produce a blank cue, and the
    # previous cue's end time should extend to cover the dropped segment.
    segments = [
        {"start": 0.0, "end": 1.92, "speaker": None, "text": "覚えていたのでしょうか"},
        {"start": 1.92, "end": 2.19, "speaker": None, "text": "。"},
    ]

    result = build_srt(segments, {})

    assert result == "1\n00:00:00,000 --> 00:00:02,190\n覚えていたのでしょうか。\n"


def test_build_srt_puts_the_speaker_prefix_on_the_first_cue_only():
    text = "今日は2つのご質問が届いています。順番にお答えしていきますね。"
    segments = [{"start": 0.0, "end": 3.1, "speaker": "SPEAKER_00", "text": text}]

    result = build_srt(segments, {"SPEAKER_00": "田中"})

    assert _cue_texts(result) == [
        "田中: 今日は2つのご質問が届いています。",
        "順番にお答えしていきますね。",
    ]


REAL_V07_SENTENCES = [
    "大高ゆうこの夢分析チャンネル。今回はリスナーからのお便り、ご質問を紹介します。",
    "でも実証されたものがとも言いません 私は10万件を超える夢を読み解いてきました",
    "夢の中の色や形は辞書に書いてある意味では読み解けません その人の深層心理に直接聞いていくしかないのです",
    "この夢主さんは、大高式ドリームワークに継続的に参加しながら、夢日記を2年間つけ続けていた方でした。",
]


@pytest.mark.parametrize("text", REAL_V07_SENTENCES)
def test_build_srt_keeps_every_cue_within_25_characters_and_loses_no_text(text):
    # The 4-character prefix "田中: " counts toward the first line's 25.
    segments = [{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00", "text": text}]

    result = build_srt(segments, {"SPEAKER_00": "田中"})

    cues = _cue_texts(result)
    assert all(len(cue) <= 25 for cue in cues), cues
    assert all(cue[0] not in "。、！？" for cue in cues), cues
    # Only whitespace separators are dropped; every other character survives.
    assert "".join(cues).replace("田中: ", "", 1) == text.replace(" ", "")


def test_build_srt_does_not_wrap_short_lines():
    segments = [{"start": 0.0, "end": 2.0, "speaker": None, "text": "短い文"}]
    result = build_srt(segments, {})
    assert result == "1\n00:00:00,000 --> 00:00:02,000\n短い文\n"


def test_build_vtt_format():
    result = build_vtt(SEGMENTS_WITH_SPEAKERS, SPEAKER_NAMES)
    assert result == (
        "WEBVTT\n\n"
        "00:00:05.000 --> 00:00:07.000\n田中: おはようございます。\n\n"
        "00:00:08.000 --> 00:00:10.000\n鈴木: よろしくお願いします。\n"
    )


def test_export_writes_requested_formats(tmp_path):
    base_path = tmp_path / "video"
    written = export(
        SEGMENTS_WITH_SPEAKERS, SPEAKER_NAMES, ["txt", "srt", "vtt"], True, str(base_path)
    )

    assert written == [
        str(base_path.with_suffix(".txt")),
        str(base_path.with_suffix(".srt")),
        str(base_path.with_suffix(".vtt")),
    ]
    for path in written:
        assert Path(path).exists()
    assert "田中" in Path(written[0]).read_text(encoding="utf-8")


def test_export_only_writes_selected_formats(tmp_path):
    base_path = tmp_path / "video"
    written = export(SEGMENTS_WITH_SPEAKERS, SPEAKER_NAMES, ["txt"], True, str(base_path))

    assert written == [str(base_path.with_suffix(".txt"))]
    assert not base_path.with_suffix(".srt").exists()


def test_export_handles_dotted_base_filename(tmp_path):
    base_path = tmp_path / "2026.09.08 meeting"
    written = export(
        SEGMENTS_WITH_SPEAKERS, SPEAKER_NAMES, ["txt", "srt", "vtt"], True, str(base_path)
    )

    assert written == [
        str(tmp_path / "2026.09.08 meeting.txt"),
        str(tmp_path / "2026.09.08 meeting.srt"),
        str(tmp_path / "2026.09.08 meeting.vtt"),
    ]
    for path in written:
        assert Path(path).exists()
