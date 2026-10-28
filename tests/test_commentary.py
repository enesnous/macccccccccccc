from macklip.commentary import clips_from_commentary
from macklip.merge import merge_clips
from macklip.models import Clip, Word


def words_at(start: float, text: str) -> list[Word]:
    return [Word(start + i * 0.4, start + (i + 1) * 0.4, t) for i, t in enumerate(text.split())]


def categories(text_by_time: dict[float, str], player="Arda Güler", aliases=("Arda",)):
    words = [w for t, txt in sorted(text_by_time.items()) for w in words_at(t, txt)]
    return [c.category for c in clips_from_commentary(words, player, list(aliases), "mac.mp4")]


def test_goal():
    assert categories({100: "Arda Güler vurdu ve GOOOL!"}) == ["gol"]


def test_assist_when_name_followed_by_pass():
    assert categories({100: "Arda Güler'in pasında Kerem Aktürkoğlu vurdu ve gol!"}) == ["asist"]


def test_other_players_goal_not_credited():
    assert categories({100: "Arda topu verdi, Kerem Aktürkoğlu vurdu, gol!"}) == ["aksiyon"]


def test_foul_on_player():
    words = words_at(50, "Arda Güler'e faul, hakem kartını çıkardı")
    clip = clips_from_commentary(words, "Arda Güler", [], "mac.mp4")[0]
    assert clip.category == "faul" and "oyuncuya yapılan" in clip.reasons[0]


def test_dribble_and_suffix_forms():
    assert categories({10: "Gülerden müthiş bir çalım"}, aliases=()) == ["çalım"]


def test_no_mention_no_clip():
    assert categories({10: "Kerem Aktürkoğlu şut çekti"}) == []


def test_merge_keeps_different_events_apart():
    a = Clip("m", 0, 20, "faul", 0.7, ["anlatım"])
    b = Clip("m", 18, 40, "gol", 0.9, ["anlatım"])
    merged = merge_clips([a, b])
    assert [c.category for c in merged] == ["faul", "gol"]
    assert merged[0].end == merged[1].start == 19


def test_merge_combines_same_moment_from_two_signals():
    a = Clip("m", 10, 20, "yakın çekim", 0.8, ["yüz"])
    b = Clip("m", 8, 22, "gol", 0.9, ["anlatım"])
    (merged,) = merge_clips([a, b])
    assert merged.category == "gol" and merged.confidence > 0.9
