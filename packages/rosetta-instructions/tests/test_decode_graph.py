"""Unit tests for decode_graph routing and node logic — no Ollama required."""

from unittest.mock import MagicMock, patch

from rosetta_schemas.models import InstructionDef, ISAMeta

from rosetta_instructions.decode_graph import (
    DecodeState,
    _STALL_LIMIT,
    _route_cursor,
    build_decode_graph,
)


# ---------------------------------------------------------------------------
# _route_cursor
# ---------------------------------------------------------------------------

def test_route_cursor_fill_when_current_set():
    state: DecodeState = {
        "current": "ADD",
        "seen": [],
        "iterations": 0,
        "stall_count": 0,
    }
    assert _route_cursor(state) == "fill"


def test_route_cursor_end_when_current_none():
    from langgraph.graph import END
    state: DecodeState = {"current": None, "seen": [], "iterations": 0, "stall_count": 0}
    assert _route_cursor(state) == END


def test_route_cursor_end_at_max_iterations():
    from langgraph.graph import END
    state: DecodeState = {
        "current": "ADD",
        "seen": [],
        "iterations": 10,
        "stall_count": 0,
        "max_iterations": 10,
    }
    assert _route_cursor(state) == END


def test_route_cursor_continues_below_max():
    state: DecodeState = {
        "current": "ADD",
        "seen": [],
        "iterations": 9,
        "stall_count": 0,
        "max_iterations": 10,
    }
    assert _route_cursor(state) == "fill"


def test_route_cursor_end_on_stall_limit():
    from langgraph.graph import END
    state: DecodeState = {
        "current": "ADD",
        "seen": ["ADD"],
        "iterations": 3,
        "stall_count": _STALL_LIMIT,
        "max_iterations": None,
    }
    assert _route_cursor(state) == END


def test_route_cursor_no_stall_below_limit():
    state: DecodeState = {
        "current": "ADD",
        "seen": ["ADD"],
        "iterations": 3,
        # _route_cursor increments stall internally before comparing, so use < LIMIT - 1.
        "stall_count": _STALL_LIMIT - 2,
        "max_iterations": None,
    }
    assert _route_cursor(state) == "fill"


# ---------------------------------------------------------------------------
# validate_node: safe-stub downgrade (tests call validate_and_fix directly)
# ---------------------------------------------------------------------------

def test_validate_downgrades_bad_pcode():
    from rosetta_instructions.validate import validate_and_fix

    instr = InstructionDef(
        mnemonic="ADD",
        encoding_bits=32,
        semantics="test",
        pcode_hint="this is not valid pcode at all",
    )
    fixed, issues = validate_and_fix(instr)
    assert fixed.pcode_hint.endswith(";")
    assert issues


def test_validate_fixes_zero_encoding_bits():
    from rosetta_instructions.validate import validate_and_fix

    instr = InstructionDef(mnemonic="NOP", encoding_bits=0, semantics="nop")
    fixed, issues = validate_and_fix(instr)
    assert fixed.encoding_bits == 32
    assert issues


def test_validate_prunes_non_binary_constraints():
    from rosetta_instructions.validate import validate_and_fix

    instr = InstructionDef(
        mnemonic="ADD",
        encoding_bits=32,
        semantics="test",
        bit_constraints={"op": "0101", "bad": "0xAB"},
    )
    fixed, issues = validate_and_fix(instr)
    assert "op" in fixed.bit_constraints
    assert "bad" not in fixed.bit_constraints
    assert issues


def test_validate_clean_instr_no_issues():
    from rosetta_instructions.validate import validate_and_fix

    instr = InstructionDef(
        mnemonic="ADD",
        encoding_bits=32,
        semantics="Add registers",
        pcode_hint="R0 = R1 + R2;",
        bit_constraints={"op": "0101"},
    )
    _, issues = validate_and_fix(instr)
    assert issues == []


# ---------------------------------------------------------------------------
# build_decode_graph: full smoke test
# ---------------------------------------------------------------------------

def _make_meta(encoding_style: str = "fixed_word") -> dict:
    return ISAMeta(
        name="TestISA",
        endian="little",
        word_size_bits=32,
        alignment=4,
        instruction_sizes_bits=[32],
        encoding_style=encoding_style,
    ).model_dump()


def test_build_decode_graph_two_instructions(tmp_path):
    """Smoke test: two-instruction run with mocked discovery and gather.

    Patches the leaf functions so the full graph wiring is exercised without
    calling an LLM.  discover returns ADD, SUB, then None to terminate.
    """
    call_count = {"n": 0}

    def fake_discover(last, seen, settings):
        n = call_count["n"]
        call_count["n"] += 1
        if n == 0:
            return "ADD", "SUB"
        if n == 1:
            return "SUB", None
        return None, None  # terminate

    def fake_gather(current, next_, settings):
        return InstructionDef(
            mnemonic=current,
            encoding_bits=32,
            semantics=f"{current} semantics",
            pcode_hint="local tmp:4 = 0;",
        )

    writer = MagicMock()
    writer.lang_dir = tmp_path / "lang"
    writer.lang_dir.mkdir(parents=True)

    settings = MagicMock()
    settings.db_path = "/tmp/db"

    with (
        patch("rosetta_instructions.discovery.discover_next", side_effect=fake_discover),
        patch("rosetta_instructions.gather.gather_instruction", side_effect=fake_gather),
        patch("rosetta_instructions.gather.enrich_pcode", side_effect=lambda i, s: i),
    ):
        app = build_decode_graph(writer)
        initial: DecodeState = {
            "settings": settings,
            "meta": _make_meta(),
            "registers": [],
            "out_dir": str(tmp_path),
            "processor_name": "TestISA",
            "max_iterations": 10,
            "inter_chunk_sleep": 0.0,
            "debug_save_dir": None,
            "resume": False,
            "last": None,
            "seen": [],
            "current": None,
            "next": None,
            "iterations": 0,
            "stall_count": 0,
            "current_def": None,
            "written": [],
            "errors": [],
        }
        final = app.invoke(initial)

    assert "ADD" in final["written"]
    assert "SUB" in final["written"]
    assert [e for e in (final.get("errors") or []) if "subgraph" in e.lower()] == []


def test_build_decode_graph_terminates_on_first_none(tmp_path):
    """If discover returns (None, None) immediately, no instructions are emitted."""

    writer = MagicMock()
    writer.lang_dir = tmp_path / "lang"
    writer.lang_dir.mkdir(parents=True)
    settings = MagicMock()

    with (
        patch("rosetta_instructions.discovery.discover_next", return_value=(None, None)),
        patch("rosetta_instructions.gather.gather_instruction"),
        patch("rosetta_instructions.gather.enrich_pcode", side_effect=lambda i, s: i),
    ):
        app = build_decode_graph(writer)
        initial: DecodeState = {
            "settings": settings,
            "meta": _make_meta(),
            "registers": [],
            "out_dir": str(tmp_path),
            "processor_name": "TestISA",
            "max_iterations": None,
            "inter_chunk_sleep": 0.0,
            "debug_save_dir": None,
            "resume": False,
            "last": None,
            "seen": [],
            "current": None,
            "next": None,
            "iterations": 0,
            "stall_count": 0,
            "current_def": None,
            "written": [],
            "errors": [],
        }
        final = app.invoke(initial)

    assert final["written"] == []
    writer.write_instruction.assert_not_called()
