"""Build and write transcript output files in TXT, SRT, and VTT formats."""
from pathlib import Path

from janome.tokenizer import Tokenizer

_tokenizer = Tokenizer()


def format_timestamp_txt(seconds: float) -> str:
    """Format seconds as HH:MM:SS for bracketed TXT timestamps."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT."""
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for WebVTT."""
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _speaker_prefix(segment, speaker_names):
    label = segment.get("speaker")
    if label is None:
        return ""
    name = speaker_names.get(label, label) if speaker_names else label
    return f"{name}: "


SRT_LINE_WIDTH = 25

_SRT_PUNCTUATION = "。、！？"
_SENTENCE_END_MARKS = "。！？!?"
_CLOSING_SYMBOLS = "」』）)】"
_COMMA_MARKS = "、,，"

# Costs (lower is better) for choosing where to split a sentence that is too
# long for one line; see _boundary_penalty and _balanced_lines.
_LINE_COUNT_PENALTY = 0.15
_OVERFLOW_PENALTY = 10.0
_MID_WORD_PENALTY = 2.0
_SHORT_LINE_LENGTH = 6  # avoid orphan lines such as a lone "分野です"
_SHORT_LINE_PENALTY = 0.6


def _pos(token):
    parts = token.part_of_speech.split(",")
    return parts[0], parts[1] if len(parts) > 1 else ""


def _is_dependent(token, previous):
    """True for tokens that attach to the word before them (particles,
    auxiliary verbs, suffixes, formal nouns such as こと, punctuation), and
    so never begin a new phrase (文節).
    """
    pos1, pos2 = _pos(token)
    if pos1 in ("助詞", "助動詞", "記号"):
        return True
    if pos1 in ("名詞", "動詞", "形容詞") and pos2 in ("接尾", "非自立"):
        return True
    # The 万 of 10万: a number continuing a number.
    return pos1 == "名詞" and pos2 == "数" and previous is not None and _pos(previous)[1] == "数"


def _starts_new_clause(token):
    return not token.surface.isspace() and not _is_dependent(token, None)


def _consists_of(surface, characters):
    return bool(surface) and all(char in characters for char in surface)


def _space_joins_words(previous, following):
    """A space between two ASCII alphanumerics ("iPhone 15") is part of the
    text; any other space is a sentence separator that Whisper emits where
    it leaves out 。.
    """
    if previous is None or following is None:
        return False
    before, after = previous.surface[-1], following.surface[0]
    return before.isascii() and before.isalnum() and after.isascii() and after.isalnum()


def _ends_sentence_without_punctuation(token, previous, following):
    """Detect a sentence end where Whisper left out 。: a sentence-final
    particle (ね, よ ...) or a polite ending (です, ます, ました, でした,
    ません) directly followed by a word that starts a new clause.
    """
    if following is None or not _starts_new_clause(following):
        return False
    pos1, pos2 = _pos(token)
    if pos1 == "助詞":
        return pos2 == "終助詞"
    if pos1 == "動詞":
        return token.surface == "ください"
    if pos1 != "助動詞":
        return False
    previous_surface = previous.surface if previous is not None else ""
    return (
        token.surface in ("です", "ます")
        or (token.surface == "た" and previous_surface in ("まし", "でし"))
        or (token.surface == "ん" and previous_surface == "ませ")
        or (token.surface == "う" and previous_surface in ("ましょ", "でしょ"))
    )


def _split_into_sentences(tokens):
    """Group tokens into sentences, ending one at 。！？, at a space, or at a
    punctuation-less sentence end. Separator spaces are dropped.
    """
    sentences = []
    current = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        previous = tokens[i - 1] if i > 0 else None
        following = tokens[i + 1] if i + 1 < len(tokens) else None

        if token.surface.isspace():
            if _space_joins_words(previous, following):
                current.append(token)
            elif current:
                sentences.append(current)
                current = []
        else:
            current.append(token)
            if _consists_of(token.surface, _SENTENCE_END_MARKS):
                while i + 1 < len(tokens) and _consists_of(
                    tokens[i + 1].surface, _SENTENCE_END_MARKS + _CLOSING_SYMBOLS
                ):
                    i += 1
                    current.append(tokens[i])
                sentences.append(current)
                current = []
            elif _ends_sentence_without_punctuation(token, previous, following):
                sentences.append(current)
                current = []
        i += 1
    if current:
        sentences.append(current)
    return sentences


def _group_into_phrases(tokens):
    """Group a sentence's tokens into phrases (文節-like units): an
    independent word plus the particles/auxiliaries/suffixes that follow it,
    with a prefix (お, ご) kept together with the word it precedes.
    """
    phrases = []
    for i, token in enumerate(tokens):
        previous = tokens[i - 1] if i > 0 else None
        # A phrase never continues past a comma, even if the next word is
        # tagged as a suffix (Janome does that to the 後 of "、後から").
        continues = phrases and previous.surface not in _COMMA_MARKS
        if continues and (_is_dependent(token, previous) or _pos(previous)[0] == "接頭詞"):
            phrases[-1].append(token)
        else:
            phrases.append([token])
    return phrases


def _boundary_penalty(last, following):
    """How undesirable it is to start a new line right after a phrase whose
    last token is `last`, before the phrase that begins with `following`.
    """
    pos1, pos2 = _pos(last)
    following_pos1 = _pos(following)[0]
    if last.surface in _COMMA_MARKS:
        return 0.0
    if pos1 == "助詞":
        if pos2 == "接続助詞":
            return 0.0
        if last.surface == "の" or pos2 in ("連体化", "並立助詞"):
            return 1.0  # modifies or joins the word that follows
        penalty = 0.5 if pos2 == "係助詞" else 0.2  # topic marker: keep with predicate
        if following_pos1 == "動詞":
            penalty += 0.6  # an argument cut off from its own verb
        return penalty
    if pos1 in ("副詞", "連体詞", "接続詞", "接頭詞"):
        return 1.0  # leans on the word that follows
    penalty = 0.4
    if pos1 == "名詞" and following_pos1 == "名詞":
        penalty += 0.5  # probably a compound noun
    if pos1 in ("動詞", "形容詞", "助動詞") and following_pos1 == "名詞":
        penalty += 0.3  # a modifier clause cut off from the noun it modifies
    return penalty


def _phrase_records(phrases, width):
    """(text, penalty for breaking after it) per phrase. A phrase longer than
    `width` has no natural break inside, so it is cut into even pieces.
    """
    records = []
    for index, phrase in enumerate(phrases):
        text = "".join(token.surface for token in phrase)
        following = phrases[index + 1][0] if index + 1 < len(phrases) else None
        penalty = _boundary_penalty(phrase[-1], following) if following is not None else 0.0
        if len(text) <= width:
            records.append((text, penalty))
            continue
        pieces = -(-len(text) // width)
        size = -(-len(text) // pieces)
        chunks = [text[i : i + size] for i in range(0, len(text), size)]
        records.extend((chunk, _MID_WORD_PENALTY) for chunk in chunks[:-1])
        records.append((chunks[-1], penalty))
    return records


def _balanced_lines(records, width):
    """Split phrase records into lines of at most `width` characters, at the
    breaks that best balance the line lengths, avoid unnatural break points
    (_boundary_penalty) and keep the number of lines low.
    """
    count = len(records)
    best = [float("inf")] * (count + 1)
    break_before = [0] * (count + 1)
    best[0] = 0.0
    for end in range(1, count + 1):
        length = 0
        for start in range(end - 1, -1, -1):
            length += len(records[start][0])
            if length > width and start != end - 1:
                break
            if length <= width:
                slack = ((width - length) / width) ** 2
            else:
                slack = _OVERFLOW_PENALTY + (length - width)
            cost = best[start] + slack + _LINE_COUNT_PENALTY
            if length < _SHORT_LINE_LENGTH:
                cost += _SHORT_LINE_PENALTY
            if end < count:
                cost += records[end - 1][1]
            if cost < best[end]:
                best[end] = cost
                break_before[end] = start

    lines = []
    end = count
    while end > 0:
        start = break_before[end]
        lines.append("".join(text for text, _ in records[start:end]))
        end = start
    lines.reverse()
    return lines


def _wrap_srt_text(text, width=SRT_LINE_WIDTH, prefix=""):
    """Split `text` into SRT lines of at most `width` characters.

    A sentence end always ends a line, so a line never carries the start of
    the next sentence. A sentence that fits on one line stays whole; a longer
    one is split between phrases (never inside one), balancing the lengths
    and preferring natural break points. `prefix` (the speaker label) is
    glued to the start of the first sentence and counts toward its width.
    """
    sentences = _split_into_sentences(list(_tokenizer.tokenize(text)))
    if not sentences:
        return [prefix] if prefix else [""]

    lines = []
    for index, sentence in enumerate(sentences):
        records = _phrase_records(_group_into_phrases(sentence), width)
        if index == 0 and prefix:
            records[0] = (prefix + records[0][0], records[0][1])
        if sum(len(record_text) for record_text, _ in records) <= width:
            lines.append("".join(record_text for record_text, _ in records))
        else:
            lines.extend(_balanced_lines(records, width))
    return lines


def _pull_leading_punctuation_across_segments(segments):
    """Move a segment's leading punctuation mark(s) onto the end of the
    previous segment's text.

    Whisper's own segmentation sometimes places a stray comma or period at
    the very start of a new segment instead of the end of the previous one
    (e.g. segment 1 ends "...時は" and segment 2 begins "、目覚めて...").
    This runs before any line-wrapping, since the within-segment wrap fix
    (_pull_leading_punctuation_to_previous_line) only reorders lines
    produced from a single segment's own text and can't reach back into a
    different segment.

    If a segment's text becomes empty after its leading punctuation is
    removed (a segment that was nothing but a punctuation mark), that
    segment is dropped and the previous segment's end time is extended to
    cover it, so no time range is lost.
    """
    result = [dict(seg) for seg in segments]
    i = 1
    while i < len(result):
        text = result[i]["text"]
        if text and text[0] in _SRT_PUNCTUATION:
            result[i - 1]["text"] += text[0]
            result[i]["text"] = text[1:]
            if result[i]["text"] == "":
                result[i - 1]["end"] = result[i]["end"]
                del result[i]
                continue
        else:
            i += 1
    return result


def _split_segment_into_srt_cues(prefix, text, start, end):
    """Split one merged segment into (start, end, line) SRT cues.

    Each line from _wrap_srt_text becomes its own cue. The segment's time
    range is divided across the cues proportionally to each line's
    character count, so longer lines get proportionally more display time.
    """
    lines = _wrap_srt_text(text, prefix=prefix)
    total_chars = sum(len(line) for line in lines) or 1
    duration = end - start
    cues = []
    cursor = start
    last_index = len(lines) - 1
    for i, line in enumerate(lines):
        if i == last_index:
            cue_end = end
        else:
            cue_end = cursor + duration * (len(line) / total_chars)
        cues.append((cursor, cue_end, line))
        cursor = cue_end
    return cues


def build_txt(segments, speaker_names, include_timestamps):
    """Build TXT content from merged segments."""
    lines = []
    for seg in segments:
        prefix = _speaker_prefix(seg, speaker_names)
        if include_timestamps:
            ts = format_timestamp_txt(seg["start"])
            lines.append(f"[{ts}] {prefix}{seg['text']}")
        else:
            lines.append(f"{prefix}{seg['text']}")
    return "\n".join(lines) + "\n"


def build_srt(segments, speaker_names):
    """Build SRT content from merged segments.

    Each segment may produce multiple cues (one per wrapped line, see
    _split_segment_into_srt_cues); cue numbers are sequential across the
    whole file, not reset per segment.
    """
    segments = _pull_leading_punctuation_across_segments(segments)
    blocks = []
    index = 1
    for seg in segments:
        prefix = _speaker_prefix(seg, speaker_names)
        cues = _split_segment_into_srt_cues(prefix, seg["text"], seg["start"], seg["end"])
        for start, end, line in cues:
            start_ts = format_timestamp_srt(start)
            end_ts = format_timestamp_srt(end)
            blocks.append(f"{index}\n{start_ts} --> {end_ts}\n{line}\n")
            index += 1
    return "\n".join(blocks)


def build_vtt(segments, speaker_names):
    """Build WebVTT content from merged segments."""
    blocks = ["WEBVTT\n"]
    for seg in segments:
        prefix = _speaker_prefix(seg, speaker_names)
        start_ts = format_timestamp_vtt(seg["start"])
        end_ts = format_timestamp_vtt(seg["end"])
        blocks.append(f"{start_ts} --> {end_ts}\n{prefix}{seg['text']}\n")
    return "\n".join(blocks)


_BUILDERS = {
    "srt": lambda segments, speaker_names, include_timestamps: build_srt(segments, speaker_names),
    "vtt": lambda segments, speaker_names, include_timestamps: build_vtt(segments, speaker_names),
    "txt": lambda segments, speaker_names, include_timestamps: build_txt(
        segments, speaker_names, include_timestamps
    ),
}


def export(segments, speaker_names, output_formats, include_timestamps, output_base_path):
    """Write transcript files for each requested format.

    Args:
        segments: merged segments, as returned by merger.merge_segments.
        speaker_names: dict mapping speaker label (e.g. "SPEAKER_00") to a
            real name. May be empty or None if no renaming was done.
        output_formats: iterable of "txt", "srt", "vtt" (any subset).
        include_timestamps: whether TXT output includes bracketed timestamps.
        output_base_path: path (without extension) to write output files to.

    Returns:
        list of file paths written, one per requested format, in the order
        given by output_formats.
    """
    speaker_names = speaker_names or {}
    written = []
    base = Path(output_base_path)
    for fmt in output_formats:
        content = _BUILDERS[fmt](segments, speaker_names, include_timestamps)
        out_path = base.with_name(base.name + f".{fmt}")
        out_path.write_text(content, encoding="utf-8")
        written.append(str(out_path))
    return written
